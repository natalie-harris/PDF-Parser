"""
Natalie Harris, NIMBioS
5/23/23

This program extracts text from downloaded pdfs, feeds it and a system_message to openai api, and retrieves information parsed by chatgpt

"""

import os
import platform
import traceback
from pdfminer.high_level import extract_text
import time
import socket
import re
import openai
import tiktoken
import glob
import shlex
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.adapters import AdapterHTTPError
from geopy.exc import GeocoderUnavailable
from geopy.exc import GeocoderServiceError
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic

import pytesseract
from PIL import Image
import pdf2image

def get_tokenized_length(text, model, examples=[]):
    """
    Calculate the number of tokens that a text string will be tokenized into 
    by a specific model. Optionally, additional content can be appended to the 
    text from a list of example dictionaries.
    
    Parameters:
    text (str): The input text string to be tokenized.
    model (str): The name or identifier of the model whose tokenizer will be used.
    examples (list of dict, optional): A list of dictionaries where each dictionary 
                                       should have a key "content" with text to append 
                                       to the input text string. Defaults to an empty list.
    
    Returns:
    int: The number of tokens the input text (plus additional content, if provided) 
         is tokenized into by the specified model.
    """
    
    # Loop through the list of example dictionaries (if provided)
    # and append the content of each example to the input text.
    for example in examples:
        text += example["content"]
    
    # Get the encoding (tokenizer) associated with the specified model.
    encoding = tiktoken.encoding_for_model(model)
    
    # Use the encoding (tokenizer) to tokenize the text
    # and then calculate the number of tokens in the tokenized text.
    num_tokens = len(encoding.encode(text))
    
    # Return the number of tokens in the tokenized text.
    return num_tokens

def get_chatgpt_response(system_message, user_message, temp=0, use_gpt4=False, examples=[], timeout=15):
    """
    Get a response from ChatGPT based on the user and system messages.

    Parameters:
    - system_message (str): The system message to set the behavior of the chat model.
    - user_message (str): The message from the user that the model will respond to.
    - temp (float, optional): Controls the randomness of the model's output (default is 0).
    - use_gpt4 (bool, optional): Flag to use GPT-4 model (default is False).
    - examples (list, optional): Additional example messages for training the model (default is an empty list).

    Returns:
    - str: The generated response from the GPT model.
    """
    
    # just to make sure gpt4 isn't used
    use_gpt4 = False

    # Combine the system and user messages to evaluate their total tokenized length
    total_message = system_message + user_message
    
    # Select the appropriate GPT model based on the use_gpt4 flag and tokenized length
    if use_gpt4:
        num_tokens = get_tokenized_length(total_message, 'gpt-4', examples)
        gpt_model = 'gpt-4'
    else:
        num_tokens = get_tokenized_length(total_message, 'gpt-3.5-turbo', examples)
        gpt_model = 'gpt-3.5-turbo' if num_tokens < 4096 else 'gpt-3.5-turbo-16k'
    
    # Prepare the messages to send to the Chat API
    new_messages = [{"role": "system", "content": system_message}]
    if len(examples) > 0:
        new_messages.extend(examples)
    new_messages.append({"role": "user", "content": user_message})
    
    # Flag to indicate whether a response has been successfully generated
    got_response = False
    
    # Continue trying until a response is generated
    retries = 0
    max_retries = 10
    while not got_response and retries < max_retries:
        try:
            # Attempt to get a response from the GPT model
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=new_messages,
                temperature=temp,
                request_timeout=timeout
            )
            
            # Extract the generated text from the API response
            generated_text = response['choices'][0]['message']['content']
            got_response = True
            return generated_text
            
        except openai.error.RateLimitError as err:
            # Handle rate limit errors
            if 'You exceeded your current quota' in str(err):
                print("You've exceeded your current billing quota. Go check on that!")
                end_runtime()  # Function to end the current runtime
            num_seconds = 3
            print(f"Waiting {num_seconds} seconds due to high volume of {gpt_model} users.")
            time.sleep(num_seconds)
            
        except openai.error.APIError as err:
            # Handle generic API errors
            print("An error occurred. Retrying request.")
            
        except openai.error.Timeout as err:
            # Handle request timeouts
            num_seconds = 3
            print(f"Request timed out. Waiting {num_seconds} seconds and retrying...")
            retries += 1
            time.sleep(num_seconds)
            
        except openai.error.ServiceUnavailableError as err:
            # Handle service unavailability errors
            num_seconds = 3
            print(f"Server overloaded. Waiting {num_seconds} seconds and retrying request.")
            time.sleep(num_seconds)

    return None

def get_latitude_longitude(town, state, country):
    """
    Get the latitude and longitude of a given location using the Nominatim geolocator.
    
    Parameters:
    - town (str): The name of the town/city.
    - state (str): The name of the state.
    - country (str): The name of the country.
    
    Returns:
    - tuple: (latitude, longitude) of the given location, or (None, None) if not found.
    """
    
    valid_sources, outbreak_occurence_values, use_gpt4, results_folder, user_agent = set_metadata()

    # Initialize the geolocator with a custom user_agent
    geolocator = Nominatim(user_agent=user_agent)
    
    # Flag to indicate if the coordinates have been found
    has_coords = False
    
    # Counter for the number of retries
    retries = 0
    num_seconds = 10
    max_attempts = 5

    # Container for the location object that will be returned by the geolocator
    location = None
    
    while not has_coords:
        try:
            # Introduce a delay before making a request to respect usage limits
            time.sleep(2)
            
            # Attempt to geolocate the query based on town, state, and country
            location = geolocator.geocode(query=f"{town}, {state}, {country}")
            has_coords = True
            
        except (AdapterHTTPError, socket.timeout) as e:
            # Handle HTTP and timeout errors
            if retries > max_attempts:
                print(f"Attempted {retries} retries. Moving on...")
                end_runtime()  # Function to end the current runtime (assumed to be defined elsewhere)
                return None, None
            
            print(f"Attempted {retries} retries. Waiting {num_seconds} seconds and trying again...")
            time.sleep(num_seconds)
            
            print("There was an HTTP error getting coordinates. Retrying...")
            
        except GeocoderUnavailable as e:
            # Handle geolocator unavailability errors
            retries += 1
            if retries > max_attempts:
                print(f"Attempted {retries} retries. Moving on...")
                end_runtime()
                return None, None
            print(f"Attempted {retries} retries. Waiting {num_seconds} seconds and trying again...")
            time.sleep(num_seconds)
            
        except GeocoderServiceError as e:
            # Handle geolocator service errors
            print("Geopy error. Moving on...")
            end_runtime()
            return None, None
        
    # If a location is found, return its latitude and longitude
    if location is not None:
        return location.latitude, location.longitude
    else:
        # If the location is not found, return None for both latitude and longitude
        return None, None

def location_to_coordinates(location, system_message):
    """
    Convert a location name to latitude and longitude coordinates using ChatGPT to
    resolve the location to a more specific city, state, and country format,
    and then use a geolocator to fetch the coordinates.
    
    Parameters:
    - location (str): A general name of a location, e.g., 'California'
    - system_message (str): The system message to set the behavior of the ChatGPT model.
    
    Returns:
    - tuple: (latitude, longitude) of the given location, or (None, None) if not found or invalid.
    """
    
    # Construct the user message by appending the specific context ('The boreal forest in') to the location
    user_message = 'The boreal forest in ' + location
    
    # Set the temperature parameter for the ChatGPT response (controls randomness)
    temperature = 0
    
    # Debugging print statement
    print("location to coords")
    
    # Get a more specific location (e.g., "city, state, country") from ChatGPT
    generated_text = get_chatgpt_response(system_message, user_message, temperature)
    if generated_text is None:
        return None, None
    
    # Debugging print statement for the generated text
    print(f"More specific location: {generated_text}")
    
    # Process the ChatGPT response: 
    # 1. Convert to lowercase
    # 2. Remove leading/trailing whitespace
    # 3. Split it into parts based on the comma separator
    generated_text = generated_text.lower().strip().split(',')
    
    # Check that the response has the expected format ("city, state, country")
    if len(generated_text) != 3:
        # If not, return None for both latitude and longitude
        return None, None
    
    # Extract the city, state, and country from the response
    city = generated_text[0].strip()
    state = generated_text[1].strip()
    country = generated_text[2].strip()
    
    # Fetch the latitude and longitude for the specific location using a geolocator function
    latitude, longitude = get_latitude_longitude(city, state, country)
    
    # Return the latitude and longitude
    return latitude, longitude

def get_location_by_coordinates(lat, long):
    """
    Get the state name for a given latitude and longitude using the Nominatim geolocator.
    
    Parameters:
    - lat (float): The latitude of the location.
    - long (float): The longitude of the location.
    
    Returns:
    - str: The state name of the given coordinates, or an empty string if not found.
    """
    
    # Initialize the geolocator with a custom user_agent
    geolocator = Nominatim(user_agent="research_paper_parser mzg857@vols.utk.edu")
    
    # Introduce a delay before making a request to respect usage limits
    time.sleep(1)
    
    # Counter for the number of retries
    retries = 0
    
    # Attempt to get the location up to 5 times if GeocoderUnavailable exception occurs
    while retries < 5:
        try:
            # Attempt to get the location data from the coordinates
            location = geolocator.reverse([lat, long], exactly_one=True)
            
            # Extract the 'address' field from the raw response
            address = location.raw['address']
            
            # Get the 'state' field from the address, defaulting to an empty string if not found
            state = address.get('state', "")
            
            return state
            
        except GeocoderUnavailable as e:
            print("Geocoder is unavailable. Retrying...")
            retries += 1
            
    # If 5 retries have been attempted without success, end the runtime
    end_runtime()

def split_with_quotes(line):
    """
    Splits a line into segments based on whitespace and commas, 
    preserving contents inside quotes as a single segment.
    
    Parameters:
    - line (str): The input string to be split.
    
    Returns:
    - list: Segments from the input string.
    """
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.whitespace += ','
    lexer.wordchars += '-'
    
    return list(lexer)

def make_csv_format(line):
    """
    Convert a line into CSV format: If line has more than 3 segments, last three segments are separated by commas, 
    the rest by spaces. If line has 3 or fewer segments, it returns the line as is.
    
    Parameters:
    - line (str): The input string to be formatted.
    
    Returns:
    - str: The formatted line.
    """
    split_line = split_with_quotes(line.lower().strip())
    length = len(split_line)

    if length <= 3:
        return line
    
    line = split_line[0]
    for i in range(1, length - 3):
        line += ' ' + split_line[i].strip()

    line += (", " + split_line[length - 3])
    line += (", " + split_line[length - 2])
    line += (", " + split_line[length - 1])

    return line

def list_each_year(original_line, publish_year=None):
    """
    Generate a list of outbreak data for each year within a given range of years.
    
    Parameters:
    - original_line (str): The input string containing location, year(s), and outbreak info.
    - publish_year (int, optional): The year the data was published.
    
    Returns:
    - list: A list of outbreak data for each year within the specified range, 
            or the original line if the year range is invalid.
    """
    print(f"original line: {original_line}")
    split_line = split_with_quotes(original_line)
    print(f"split_with_quotes: {split_line}")

    location = split_line[0].strip()
    years = split_line[1].strip()
    outbreak = split_line[2].strip()
    first_year = years[:4]
    last_year = years[-4:]

    # Error Handling: Check if years are digits
    if not first_year.isdigit() or (len(years) > 5 and not last_year.isdigit()):
        return [original_line]

    first_year = int(first_year)

    # Define last_year based on input format or publish_year
    if len(years) == 5 and publish_year is not None:
        last_year = int(publish_year)
    elif len(years) == 9:
        last_year = int(last_year)
    else:
        return [original_line]
    
    # Validate the year range
    if first_year >= last_year or last_year - first_year > 60 or first_year > 2022 or last_year > 2023:
        return [original_line]

    # Check against publish_year if provided
    if publish_year is not None and (first_year > publish_year - 1 or last_year > publish_year):
        return [original_line]
    
    # Generate a list of outbreak data for each year in the range
    new_list = []
    for i in range(first_year, last_year + 1):
        strings = [f'"{location}"', str(i), outbreak]
        new_line = ", ".join(strings)
        new_list.append(new_line)

    return new_list

def bp_to_bc_ad(bp_date):
    """
    Convert calibrated years before present (e.g., "5000 cal yr BP") to a regular year (BC/AD).
    
    Parameters:
    - bp_date (str): The input string containing the calibrated years before present.
    
    Returns:
    - int: The calculated regular year (BC/AD), or None if conversion fails.
    """
    try:
        # Extract numeric part of bp_date using regex and calculate the regular year
        bp_year = int(re.match(r"(\d+)\s*cal\s*yr\s*bp", bp_date.replace(' ', '').lower()).groups()[0])
        year = 1950 - bp_year
        return year
    except Exception as e:
        # Print an error message and return None if conversion fails
        print(f"Error occurred: {str(e)}")
        return None

def remove_nonnumeric_chars(input_string):
    """
    Remove non-numeric characters from the input string.
    
    Parameters:
    - input_string (str): The input string that may contain non-numeric characters.
    
    Returns:
    - str: The cleaned string containing only numeric characters.
    """
    cleaned_string = re.sub(r'[^0-9]', '', input_string)
    return cleaned_string

def clean_dates(date):
    """
    Clean date strings by removing certain redundant or unclear year signifiers.
    
    Parameters:
    - date (str): The input date string that may contain extra characters or words.
    
    Returns:
    - str: The cleaned date string.
    """
    date = str(date).replace("ca.", "")  # Remove "ca." prefix, which stands for "circa"
    date = date.replace("s", "")         # Remove 's' suffix to transform decade references (e.g., "1970s" becomes "1970")
    date = date.strip()                  # Remove leading and trailing whitespace
    return date

def parse_response(response, outbreak_df, system_message_stage_4, general_latitude=0.0, general_longitude=0.0, general_state='None', state_cache={}, publish_year=None):

    """
    Parse GPT output to convert location names to coordinates and filter data.
    
    This function takes the response from GPT and performs various clean-up operations, 
    including formatting the data, converting 'cal yr bp' to regular years, handling ranges 
    of years, checking a cache for previously found location coordinates, and updating 
    a DataFrame with the cleaned and formatted outbreak data.
    
    Args:
    - response (str): The raw output from GPT.
    - outbreak_df (DataFrame): The DataFrame to be updated with the cleaned data.
    - system_message_stage_4 (obj): A placeholder for a system message object.
    - general_latitude (float, optional): Defaults to 0.0. General latitude to use if specific coordinates are not found.
    - general_longitude (float, optional): Defaults to 0.0. General longitude to use if specific coordinates are not found.
    - general_state (str, optional): Defaults to 'None'. The general state of the location.
    - state_cache (dict, optional): A cache for storing state information to avoid redundant API calls.
    - publish_year (int, optional): The publication year of the data, used for additional filtering.
    
    Returns:
    - DataFrame: The updated outbreak DataFrame.
    - dict: The updated state cache.
    """

    split_response = response.splitlines()
    new_split_response = [] # used for adding additional lines when chatgpt gives range of years
    latitude = None
    longitude = None

    # cache previously found location coordinates to reduce chatgpt use
    cached_location_coords = {}

    # for each line in chatgpt's response
    for line in split_response:

        # make sure line is in the correct format, otherwise move to next line
        line = make_csv_format(line)
        split_line = split_with_quotes(line)
        # split_line = line.split(',')
        print(f"\n\nbeginning split line: {split_line}")

        if len(split_line) != 3:
            continue

        location = split_line[0].strip().lower().strip('"')
        year = clean_dates(split_line[1].strip().lower().strip('"'))
        outbreak = split_line[2].strip().lower().strip('"')

        print(f"Slightly more formatted: {line}\nLocation: {location}, Year: {year}, Outbreak: {outbreak}")

        if check_for_multiple_locations(location):
            print(f"{location} was determined to not be a valid location. Continuing to next line...")
            continue

        if is_unknown(location):
            print("Location was not found. Continuing...")
            continue

        if outbreak != 'yes' and outbreak != 'no' and outbreak != 'uncertain':
            continue
        if any(char.isalpha() for char in year):
            continue
        if len(year) != 4 and len(year) != 5 and len(year) != 9 and 'cal' not in year.lower():
            continue
        if len(location) <= 3:
            continue

        print("Got here") 

        # if cal yr bp is in years, convert them to regular years
        if 'cal' in year.lower():
            if year.count('-') == 1:
                before_dash, after_dash = year.split('-')
                before_dash = before_dash.strip().lower()
                after_dash = after_dash.strip().lower()

                if 'cal' in before_dash:
                    before_dash = bp_to_bc_ad(before_dash)
                if 'cal' in after_dash:
                    after_dash = bp_to_bc_ad(after_dash)

                if not str(before_dash).isnumeric() or not str(after_dash).isnumeric():
                    continue

                year = f"{before_dash}-{after_dash}"

            elif year.count('-') == 0:
                year = bp_to_bc_ad(year)
                if not str(year).isnumeric():
                    continue

            else:
                continue

        # create human check here to fix glaring errors
        check = False
        if check:
            choice = 1
            orig_location = location
            orig_year = year
            orig_outbreak = outbreak
            while choice != '0' and choice != '':
                print(f'\nLine: "{location}", "{year}", "{outbreak}"')
                choice = input('Press 0 to continue\nPress 1 to change location\nPress 2 to change year\nPress 3 to change outbreak\nPress 4 to reset to original values\nPress 5 to skip this line entirely > ')
                if choice == '1':
                    location = input("New location: ")
                elif choice == '2':
                    year = input("New year: ")
                elif choice == '3':
                    outbreak = input("New outbreak")
                elif choice == '4':
                    location = orig_location
                    year = orig_year
                    outbreak = orig_outbreak
                elif choice == '5':
                    continue

        new_line = ", ".join([f'"{location}"', year, outbreak])

        print("before range")
        # if data given as range of years, add every year to new list
        print(year)
        print('hello')
        if (len(year) == 5 or len(year) == 9) and year[4] == '-':
            every_year = list_each_year(new_line, publish_year)
            print('goodbye')
            if len(every_year) > 1:
                for single_year in every_year:
                    new_split_response.append(single_year)

        elif publish_year is not None:
            if int(year) <= publish_year:
                new_split_response.append(new_line)
        else:
            if int(year) <= 2023:
                new_split_response.append(new_line)

        print("after range")
        if len(new_split_response):
            print("Successfully split response!")
        else:
            print("Unsuccessfully split response...")
        # print(f"New Split Response: {new_split_response}")

    for line in new_split_response:
        print(f"line from NSR: {line}")

        split_line = split_with_quotes(line)
        print(split_line)
        location = split_line[0].lower().strip()
        year = split_line[1].lower().strip()
        outbreak = split_line[2].lower().strip()

        # search for location in cache, otherwise get coordinates and store in cache
        if location in cached_location_coords:
            latitude = cached_location_coords[location][0]
            longitude = cached_location_coords[location][1]
        else:
            latitude, longitude = location_to_coordinates(location, system_message_stage_4)
            print(latitude, longitude)
            if latitude == None or longitude == None:
                # need to implement replacing with coordinates if possible
                if general_latitude != 0.0 and general_longitude != 0.0:
                    latitude = general_latitude
                    longitude = general_longitude
                else:
                    return outbreak_df, state_cache
                
            cached_location_coords[location] = (latitude, longitude)
        
        new_state = ''
        # if entry is not in general location's state/province, don't add entry to df
        if (latitude, longitude) in state_cache:
            new_state = state_cache[(latitude, longitude)]
        else:
            new_state = get_location_by_coordinates(latitude, longitude).lower()
            state_cache[(latitude, longitude)] = new_state

        # if general_state != new_state:
        #     return outbreak_df, state_cache
        
        print("got here")

        # add latitude and longitude to dataframe
        split_line.append(latitude)
        split_line.append(longitude)
        print(f"{location}, {latitude}, {longitude}, {year}, {outbreak}")
        # print(outbreak_df)
        outbreak_df.loc[len(outbreak_df)] = [location, latitude, longitude, year, outbreak, '']

    return outbreak_df, state_cache

def build_chunk_group(system_message, text, end_message="\n\nEND\n\n", use_gpt4=False, examples=[], just_one_chunk=False, max_context_length=None):
    """
    Returns chunks of text that stay within a specified token limit.
    
    Args:
    - system_message (str): The message to prepend to each chunk of text.
    - text (str): The full text that needs to be split into chunks.
    - end_message (str, optional): The message to append to each chunk of text.
    - use_gpt4 (bool, optional): If true, use the token limit for GPT-4.
    - examples (list, optional): List of examples for tokenization.
    - just_one_chunk (bool, optional): If true, return only one chunk.
    - max_context_length (int, optional): If not None, use specified token limit.

    Returns:
    - list: A list of chunks, where each chunk is a list containing the system message, a segment of the text, and the end message.
    """

    # Define initial setup values
    system_message_length = len(system_message) + len(end_message)
    max_token_length = 16000  # Default max token length for GPT-3
    if use_gpt4:
        max_token_length = 8000  # GPT-4 token limit
    if max_context_length is not None and max_context_length <= max_token_length:
        max_token_length = max_context_length  # Explicit token limit
    elif max_context_length is not None:
        print(f"Specified maximum context length is too long for GPT. Using {max_token_length} instead.")
    
    base_multiplier = 4
    safety_multiplier = 0.9  # Reduce token size to avoid potential overflows due to local tokenizer differences

    chunk_group = []  # Will hold the resulting chunks of text

    i = 0  # Start index for slicing the text
    while i < len(text):

        # Calculate the length of a user message chunk
        multiplier = base_multiplier
        user_message_length = int(max_token_length * multiplier) - system_message_length

        # Build initial message
        message = system_message + text[i:i+user_message_length] + end_message

        # Assume 'get_tokenized_length' is a function that returns the token count of a message
        token_length = get_tokenized_length(message, 'gpt-3.5-turbo', examples)
        
        # If the token length exceeds the max allowed, reduce the message length and recheck
        while token_length > int(max_token_length * safety_multiplier):
            multiplier *= 0.95
            user_message_length = int(max_token_length * multiplier) - system_message_length
            message = system_message + text[i:i+user_message_length] + end_message
            token_length = get_tokenized_length(message, 'gpt-3.5-turbo', examples)
        
        # Save the chunk and move to the next segment of text
        chunk_group.append([system_message, text[i:i+user_message_length] + end_message])
        i += user_message_length

        # Stop if only one chunk is needed
        if just_one_chunk:
            break

    return chunk_group

def dms_to_dd(dms):
    """
    Convert a string representing a latitude or longitude in either DD or DMS format to DD format.

    Args:
    - dms (str): A string representing a latitude or longitude in either DD or DMS format.

    Returns:
    - float: The latitude or longitude in DD format.

    Raises:
    - ValueError: If the input string is not in a valid format.
    """

    # Check if input is in decimal degree format (e.g., "45.5°N")
    match = re.match(r"[-+]?[0-9]*\.?[0-9]+°[NSWE]", dms)
    if match:
        # Extract degree value and direction (N/S/E/W) from the string
        degree, direction = re.match(r"([-+]?[0-9]*\.?[0-9]+)°([NSWE])", dms).groups()

        # Convert the degree string to float
        dd = float(degree)

        # If the direction is South (S) or West (W), make the degree value negative
        if direction in 'SW':
            dd *= -1

        return dd

    # If the input is not in DD format, assume it's in DMS format
    match = re.match(r"(\d+)°(\d+)?'?(?:([0-9.]+)?\"?)?([NSWE])?", dms)
    if match:
        # Extract degrees, minutes, seconds, and direction from the string
        degrees, minutes, seconds, direction = match.groups()

        # Convert the DMS values to DD format
        dd = float(degrees) + (float(minutes) if minutes else 0)/60 + (float(seconds) if seconds else 0)/3600

        # If the direction is South (S) or West (W), make the degree value negative
        if direction and direction in 'SW':
            dd *= -1

        return dd

    # If input string is not in a recognized format, raise a ValueError with a message
    raise ValueError("Invalid DMS or DD format. Please enter a valid format (e.g., '45.5°N' or '45°30'15\"N')")

def get_centroid_of_bb(bounding_box):
    """
    Compute the centroid of a bounding box defined by coordinates in degrees,
    minutes, and seconds (DMS) or decimal degrees (DD).
    
    Args:
    - bounding_box (str): A string describing the bounding box in the format 
      'lat1-lat2, lon1-lon2'. Coordinates may be in DMS or DD.
    
    Returns:
    - tuple: A tuple (latitude, longitude) representing the centroid of the bounding box, or
      None if computation fails after 5 attempts.
    """

    retries = 0  # Keep track of the number of attempts to compute the centroid

    while retries < 5:  # Allow up to 5 attempts
        try:
            # Print the bounding box for debugging
            print(bounding_box)

            # Extract coordinates from the bounding box string using regex
            lat1_str, lat2_str, lon1_str, lon2_str = re.match(r"(.+?)-(.+?),\s*(.+?)-(.+)", bounding_box).groups()
            
            # Convert coordinate strings from DMS to decimal degrees
            print('bounding box')
            print(f'{lat1_str} | {lat2_str} | {lon1_str} | {lon2_str}')
            lat1 = dms_to_dd(lat1_str)
            lat2 = dms_to_dd(lat2_str)
            lon1 = dms_to_dd(lon1_str)
            lon2 = dms_to_dd(lon2_str)

            # Compute the geodesic line between the two corners of the bounding box
            line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)

            # Compute the midpoint (centroid) of the geodesic line
            midpoint = line.Position(0.5 * line.s13)

            # Return the latitude and longitude of the midpoint
            return (midpoint["lat2"], midpoint["lon2"])

        except Exception as e:
            # Print an error message and retry after waiting for 1 second
            print(f"Error occurred: {str(e)}. Retrying.")
            time.sleep(1)
            retries += 1  # Increment the retry count

    # End the runtime if all attempts fail
    end_runtime()

def parse_coordinates(coordinates):
    """
    Parse a string containing latitude and longitude coordinates,
    and convert them from DMS format to decimal degrees.
    
    Args:
    - coordinates (str): A string in the format 'latitude, longitude'. 
    
    Returns:
    - tuple: A tuple (latitude, longitude) in decimal degrees, or (None, None) if parsing or conversion fails.
    """
    
    try:
        # Split input into latitude and longitude strings
        lat_str, lon_str = re.match(r"(.+),\s*(.+)", coordinates).groups()
        
        # Convert latitude and longitude from DMS to decimal degrees
        lat = dms_to_dd(lat_str)
        lon = dms_to_dd(lon_str)
        
        return lat, lon

    except Exception as e:
        # Print error and return None for both latitude and longitude
        print(f"Error occurred: {str(e)}")
        return None, None

def clean_coordinates(coordinates):
    """
    Parses various coordinate formats to extract latitude and longitude.
    
    This function checks the format of the input coordinates, processes the input
    accordingly, and returns cleaned latitude and longitude values, or None if
    parsing fails.

    Parameters:
    coordinates (str): The input string containing coordinate information.
    
    Returns:
    latitude (float or None): The parsed latitude value.
    longitude (float or None): The parsed longitude value.
    """

    latitude, longitude = None, None
    temperature = 0
    
    if not coordinates.startswith('unknown'):
        # Check if the input contains any digits
        contains_numbers = any(char.isdigit() for char in coordinates)

        if not contains_numbers:  # Skip input without any numbers
            return latitude, longitude

        # Get the classification of the coordinates format (e.g., bounding box, DMS, etc.)
        coord_classification = get_chatgpt_response(system_messages["system_message_stage_0c"], coordinates, temperature)
        if coord_classification is None:
            return None, None
        coord_classification = coord_classification.lower().strip()

        if coord_classification in ['bounding box', 'degrees/minutes', 'degrees/minutes/seconds', 'decimal degrees']:
            formatted_coords = 'unknown'

            # Parse coordinates based on their classification
            if coord_classification == 'bounding box':
                formatted_coords = get_chatgpt_response(system_messages["system_message_stage_0c_boundingbox"], coordinates, temperature)
                if formatted_coords is None:
                    return None, None
                latitude, longitude = get_centroid_of_bb(formatted_coords)
            elif coord_classification == 'degrees/minutes':
                formatted_coords = get_chatgpt_response(system_messages["system_message_stage_0c_dm"], coordinates, temperature)
                if formatted_coords is None:
                    return None, None
                latitude, longitude = parse_coordinates(formatted_coords)
            elif coord_classification == 'degrees/minutes/seconds':
                formatted_coords = get_chatgpt_response(system_messages["system_message_stage_0c_dms"], coordinates, temperature)
                if formatted_coords is None:
                    return None, None
                latitude, longitude = parse_coordinates(formatted_coords)
            elif coord_classification == 'decimal degrees':
                formatted_coords = get_chatgpt_response(system_messages["system_message_stage_0c_dd"], coordinates, temperature)
                if formatted_coords is None:
                    return None, None
                latitude, longitude = parse_coordinates(formatted_coords)

    return latitude, longitude

def extract_abstract_to_references(text):
    """
    Extracts the content of a paper from the 'abstract' section to the 'references' section.
    
    If 'abstract' or 'references' section is not found, it defaults to the beginning or end of the text, respectively.

    Parameters:
    text (str): The input string containing the entire content of the paper.
    
    Returns:
    str: The extracted content from 'abstract' to 'references'.
    """
    # Convert to lowercase for case-insensitive search
    lower_case_text = text.lower()

    try:
        # Find the first occurrence of 'abstract'
        start = lower_case_text.index('abstract')
    except ValueError:
        # If 'abstract' is not found, start from the beginning of the text
        start = 0

    # Override to always start from the beginning of the text
    start = 0

    try:
        # Find the last occurrence of 'references' and adjust to the end of the word
        end = lower_case_text.rindex('references') + len('references')
    except ValueError:
        # If 'references' is not found, end at the last character of the text
        end = len(text)

    # Extract the substring from 'abstract' to 'references' 
    # (or the beginning/end of the text if either is not found)
    extracted_text = text[start:end]
    
    return extracted_text

def cleanup_text(text):
    """
    Cleans up the input text by removing unnecessary tabs, newlines, and spaces,
    and by normalizing dashes and apostrophes.

    Parameters:
    text (str): The input string that needs to be cleaned up.
    
    Returns:
    str: The cleaned-up version of the input text.
    """
    # List of pairs where each pair contains a pattern to find and its replacement
    to_replace = [
        [' \t', ' '], 
        [' \n', ' '], 
        [' \'', '\''], 
        ['-   ', '-'], 
        ['-  ', '-'], 
        ['- ', '-'], 
        ['  ', ' '], 
        [' –', '-']
    ]

    # Iteratively replace all unwanted patterns in the text
    for pair in to_replace:
        text = text.replace(pair[0], pair[1])

    return text

def get_study_index(file, study_indices):
    """
    Retrieve the study index associated with a given file from a dictionary of study indices.
    
    Parameters:
    file (str): The full path of the file whose study index needs to be retrieved.
    study_indices (dict): A dictionary containing file names as keys and their associated study indices as values.
    
    Returns:
    int or None: The study index associated with the file, or None if the file is not found in the study_indices dictionary.
    """
    
    # Extract the file name from the full file path
    file = file.rsplit('/', 1)[-1]
    
    # Check if the file name exists in the study_indices dictionary and return the corresponding value if it does
    if file in study_indices:
        return study_indices[file]
    
    # Return None if the file name is not found in the study_indices dictionary
    return None

def get_val_from_dict(key, dict):
    # helper function that makes things a little more concise
    if key in dict:
        return dict[key]
    return None

def wait():
    # for debugging
    input("Waiting...")

def yes_or_no(response):
    # for determining the response from chatgpt on yes/no tasks
    # print(response)
    if 'yes' in response.lower():
        return True
    return False

def is_unknown(response):
    synonyms = ['unknown', 'unspecifi', 'not known', 'not understood']
    for synonym in synonyms:
        if synonym in response.lower():
            return True
    return False

def find_unused_filename(base_dir="Testing/testing_data", file_prefix="test"):
    """
    Generate a unique file name by appending an incrementing integer until an unused name is found.
    
    Parameters:
    base_dir (str): The directory where the file is intended to be saved.
    file_prefix (str): The prefix of the file name.
    
    Returns:
    str: A unique file name that does not currently exist in the specified directory.
    """
    
    current_os = get_os()

    i = 1
    while True:
        # Generate a potential file name using the given directory, prefix, and current index
        if current_os == 'Windows':
            potential_name = f"{base_dir}\{file_prefix}{i}"
        elif current_os == 'Darwin':
            potential_name = f"{base_dir}/{file_prefix}{i}"
        
        # Check if a file with the generated name already exists
        if not os.path.exists(potential_name + ".csv"):
            # If not, return the unique file name
            return potential_name
        
        # Increment the index and try the next potential name
        i += 1

def end_runtime():
    """
    Consolidate and process dataframes, and save to CSV and Excel files.
    Expects dataframes in data_list with specific columns.
    Writes output to files named according to the global variable file_name.
    Maps 'Outbreak' values using the global variable outbreak_occurence_values.
    """
    global data_list
    global file_name
    global outbreak_occurence_values
    global pdf_df

    current_os = get_os()

    if current_os == "Darwin":
        csv_filename = r'Results/all_pdfs.csv'
    elif current_os == "Windows":
        csv_filename = r'Results\all_pdfs.csv'
    else:
        print("OS not supported")
        exit()
    
    pdf_df.to_csv(csv_filename, index=False)

    print(data_list)

    final_df = []  # Initialize list to store the final processed dataframes

    # Iterate through each dataframe in data_list
    for df in data_list:
        data = df.copy()
        filename = data['File Name'].iloc[0]
        source = data['Source'].iloc[0]
        study = data['Study'].iloc[0]
        list_data_filled = []

        data = data.sort_values(['area', 'Year'])  # Sort by area and year
        
        # Iterate through each unique area in the current dataframe
        for area in data['area'].unique():
            area_data = data[data['area'] == area].copy()
            area_data['Year'] = area_data['Year'].astype(int)  # Convert 'Year' column to int
            
            min_year = int(area_data['Year'].min())
            max_year = int(area_data['Year'].max())
            latitude = area_data['Latitude'].iloc[0]
            longitude = area_data['Longitude'].iloc[0]
            
            # # Create DataFrame for all years within min and max range for this area
            # all_years = pd.DataFrame({'Year': range(min_year - 1, max_year + 2)})
            # all_years['area'] = area
            # all_years['Latitude'] = latitude
            # all_years['Longitude'] = longitude
            
            # # Merge all_years with the specific area data from area_data
            # merged_data = pd.merge(all_years, area_data, how='left', on=['Year', 'area', 'Latitude', 'Longitude'])
            # merged_data['Outbreak'].fillna('no', inplace=True)  # Fill NA 'Outbreak' with 'no'
            # merged_data['Study'] = study
            # merged_data['File Name'] = filename
            # merged_data['Source'] = source
            
            # list_data_filled.append(merged_data)  # Append processed data for this area
            list_data_filled.append(area_data)

        # Concatenate processed data of all areas
        data = pd.concat(list_data_filled, ignore_index=True)
        
        # Map 'Outbreak' column values using predefined mapping
        data['Outbreak'] = data['Outbreak'].map(outbreak_occurence_values)
        
        final_df.append(data)  # Append the fully processed dataframe to final_df list

    # Save data to CSV and Excel if there are any processed dataframes
    if len(final_df) > 0:
        all_data = pd.concat(final_df, ignore_index=True)  # Concatenate all processed dataframes
        all_data.to_csv(file_name + '.csv', index=False)  # Save to CSV
        all_data.to_excel(file_name + '.xlsx', index=False)  # Save to Excel
        print(f"Written to {file_name} (.csv and .xlsx)")  # Print confirmation

    # Exit the program
    exit()
    
def extract_text_from_pdf(pdf_path):
    """
    Extract and concatenate text content from all pages of a PDF file.
    
    Args:
    pdf_path (str): Path to the input PDF file.
    
    Returns:
    str: Concatenated text content of the PDF.
    """
    
    # Use pdfminer's extract_text function
    text = extract_text(pdf_path)
    
    return text

def extract_text_from_scanned_pdf(file_path, poppler_bin_path):
    """
    Extract and concatenate text content from all pages of a scanned PDF file.

    Args:
    file_path (str): Path to the input scanned PDF file.
    poppler_bin_path (str): Path to the Poppler binary.

    Returns:
    str: Concatenated text content of the scanned PDF.
    """
    
    print("Using OCR...")

    # Convert PDF to list of images
    images = pdf2image.convert_from_path(file_path, poppler_path=poppler_bin_path)


    # Extract text from each image using pytesseract
    texts = [pytesseract.image_to_string(img) for img in images]
    
    return "\n".join(texts)  # Join and return the extracted text from all pages

def set_metadata():
    valid_sources = [
        'dendrochronological samples from tree cores', 
        'dendrochronological samples from historical buildings',
        'dendroarcheological material', 
        'pheromone traps', 
        'aerial defoliation survey',
        'survey from insect laboratory', 
        'personal communication with the department of lands and forest representative'
    ]
    outbreak_occurence_values = {
        'no': 0,
        'yes': 1,
        'uncertain': 2
    }
    use_gpt4 = False
    results_folder = r""
    user_agent = ""
    return valid_sources, outbreak_occurence_values, use_gpt4, results_folder, user_agent

def set_ocr_metadata():

    # get global data_list var
    global data_list

    # set up OCR variables
    current_os = get_os()
    if current_os == 'Windows':
        poppler_bin_path = r'C:\Users\natal\OneDrive\Documents\GitHub\References_Finder\windows_venv\poppler-23.08.0\Library\bin'
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif current_os == 'Darwin':
        poppler_bin_path = r'/usr/local/bin/'
        tesseract_path = r'/usr/local/bin/tesseract'
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    return poppler_bin_path, tesseract_path

def read_api_key(file_name):
    with open(file_name, 'r') as file:
        api_key = file.read().strip()  # Read the content and remove any leading or trailing whitespaces
    return api_key

def process_file(file):
    print(f"\n\n\nCurrently Processing: {file}")

    global data_list
    global state_cache
    global location_coordinates

    # ChatGPT cannot read the pictures in Hardy et al. so we can't compare data
    study_indices, valid_sources, general_locations, general_coords, max_boundary_percentage, use_gpt4 = set_metadata()

    # get info for OCR analysis
    poppler_bin_path, tesseract_path = set_ocr_metadata()

    # get text from file
    pdf_text = extract_text(file)

    # if extracted text is short, try OCR
    if len(pdf_text) < 100:
        pdf_text = extract_text_from_scanned_pdf(file, poppler_bin_path)

    # remove everything before abstract and after/on references page
    pdf_text = extract_abstract_to_references(pdf_text)
    pdf_text = cleanup_text(pdf_text)

    # set up openai api
    # you must set your own openai key in new_openai_key.txt
    openai_key = read_api_key("new_openai_key.txt")
    if len(openai_key) <= 0:
        print("You must set your own openai key in new_openai_key.txt")
        end_runtime()
    openai.api_key = openai_key
    model_list = openai.Model.list()

    # Get relevance of text
    relevance_chunk = build_chunk_group(system_messages["system_message_topic_checker"], pdf_text, just_one_chunk=True, max_context_length=4096)[0][1]
    relevance_response = get_chatgpt_response(system_messages["ystem_message_topic_checker"], relevance_chunk, use_gpt4=False)
    if relevance_response is None:
        is_relevant = False
    else:
        is_relevant = yes_or_no(relevance_response)
    print(f"Identified relevance: {is_relevant}")
    if not is_relevant:
        return
 
    # get source of data
    # build chunks
    source_chunk_group = build_chunk_group(system_messages["system_message_stage_0"], pdf_text, end_message, max_context_length=4096)
    source_prefix = "data collection method: "

    # iterate through each chunk until source is found
    source = 'unknown'
    for chunk in source_chunk_group:
        system_message = chunk[0]
        user_message = chunk[1]
        temperature = 0
        generated_text = get_chatgpt_response(system_message, user_message, temperature)
        if generated_text is None:
            continue
        generated_text = generated_text.lower()
        if generated_text.startswith(source_prefix):
            generated_text = generated_text[len(source_prefix):]
        if generated_text.endswith('.'):
            generated_text = generated_text[0:len(generated_text) - 1]
        if not generated_text.startswith('unknown'):
            source = generated_text
        if not is_unknown(source):
            break

    found_valid_sources = []
    source = source.split(',')
    for i in range(len(source)):
        source[i] = source[i].strip().lower()
        if source[i] in valid_sources:
            found_valid_sources.append(source[i])
    
    print(f"Found sources: {found_valid_sources}")

    # get location of data to use in case location cannot be found
    stage0b_chunks = build_chunk_group(system_messages["system_message_stage_0b"], pdf_text, end_message, max_context_length=4096)
    location_prefix = 'location: '

    found_coordinates = False
    location = 'unknown'
    latitude, longitude = None, None
    coord_classification = ''
    i = 0

    study_index = get_study_index(file, study_indices)
    gen_location = get_val_from_dict(study_index, general_locations)
    gen_coords = get_val_from_dict(study_index, general_coords)

    if gen_coords is None:
        while i < len(stage0b_chunks) and is_unknown(location.lower()):
            chunk = stage0b_chunks[i]
            system_message = chunk[0]
            user_message = chunk[1]
            temperature = 0             

            generated_text = get_chatgpt_response(system_message, user_message, temperature)
            if generated_text is None:
                generated_text = 'unknown'
            generated_text = generated_text.lower()

            print(generated_text)

            if generated_text.startswith(location_prefix):
                generated_text = generated_text[len(location_prefix):]
            if generated_text.endswith('.'):
                generated_text = generated_text[0:len(generated_text) - 1]
            
            latitude, longitude = clean_coordinates(generated_text)
            if latitude is not None and longitude is not None:
                found_coordinates = True

            i += 1

        # if the coords are bounding boxes, we get the centroid
        if coord_classification == 'bounding box':
            latitude, longitude = get_centroid_of_bb(location)
        if latitude is not None and longitude is not None:
            print(f'\n\nBOUNDING BOX CENTROID: {latitude}, {longitude}\n')

    else:
        coord_classification = 'decimal degrees'
        found_coordinates = True
        latitude = gen_coords[0]
        longitude = gen_coords[1]


    # try to guess the year the work was published to limit years we can record data from it
    year_guess = None
    year_guesser_chunk_group = build_chunk_group(system_messages["system_message_year_guesser"], pdf_text, end_message, max_context_length=4096)
    if len(year_guesser_chunk_group) > 0:
        user_message = year_guesser_chunk_group[0][1]
    
        generated_text = get_chatgpt_response(system_messages["system_message_year_guesser"], user_message, temperature)
        
        if generated_text is not None:
            # Remove non-numeric characters from generated_text
            generated_text = re.sub(r'\D', '', generated_text)

            # Check if the resulting value is a 4 digit number
            if len(generated_text) == 4:
                year_guess = int(generated_text)
                print(f"Inferred year: {year_guess}")


    # try to find the general area of the study and get coordinates from geopy if coordinates are not already found
    if gen_location is None:

        stage0c_chunks = build_chunk_group(system_messages["system_message_stage_0d"], pdf_text, end_message, use_gpt4)
        location_prefix = 'location: '

        location = 'unknown'
        i = 0
        while i < len(stage0c_chunks) and is_unknown(location):
            chunk = stage0c_chunks[i]
            system_message = chunk[0]
            user_message = chunk[1]
            temperature = 0

            generated_text = get_chatgpt_response(system_message, user_message, temperature, use_gpt4)

            if generated_text is not None:
                print(f'Generated General Location: {generated_text}')
                generated_text = generated_text.lower()
                if generated_text.startswith(location_prefix):
                    generated_text = generated_text[len(location_prefix):]
                if generated_text.endswith('.'):
                    generated_text = generated_text[0:len(generated_text) - 1]
                if not is_unknown(generated_text):
                    location = f'"{generated_text}"'
                    if latitude is None or longitude is None:
                        if location not in location_coordinates:
                            latitude, longitude = location_to_coordinates(location, system_messages["system_message_stage_4"])
                            if latitude is not None and longitude is not None:
                                location_coordinates[location] = (latitude, longitude)
                                found_coordinates = True
                        else:
                            latitude, longitude = location_coordinates[location][0], location_coordinates[location][1]
                        if latitude is not None and longitude is not None:
                            found_coordinates = True

            i += 1
    else:
        location = gen_location

    state = None
    if found_coordinates:
        print(latitude, longitude)
        state = get_location_by_coordinates(latitude, longitude).lower()
        print(state)
    print(location)


    # if not found_coordinates, then we need to do the chatgpt process to get coordinates from geopy

    # set up dataframe for csv output
    outbreak_df = pd.DataFrame(columns=['area', 'Latitude', 'Longitude', 'Year', 'Outbreak', 'Source'])

    location_backup = ""
    if 'unknown' not in location.lower():
        location_backup = f"\n\nIn case the text doesn't provide a location for an outbreak, we've found that the outbreaks in this text take place at {location}"

    # build prompt chunks
    chunk_group = build_chunk_group(system_messages["system_message_stage_1"] + location_backup, pdf_text, end_message, use_gpt4, few_shot_examples["stage_1_few_shot_examples"], max_context_length=4196)

    stage1_results = ''

    # make api call for each chunk in each chunk_group, print response
    chunk_number = 1
    for chunk in chunk_group:
        system_message = chunk[0]
        user_message = chunk[1]
        temperature = 0

        # print(f'Text chunk: {user_message}')

        # while True:
        #     time.sleep(1)

        generated_text = get_chatgpt_response(system_message, user_message, temperature, use_gpt4)
        if generated_text is None:
            continue

        generated_text = cleanup_text(generated_text)
        stage1_results += f'\nText chunk {chunk_number}: [{generated_text}]'
        chunk_number += 1

    print(f"\nStage 1: {stage1_results}\n\n")

    generated_text = get_chatgpt_response(system_messages["system_message_stage_2"], stage1_results, 0, use_gpt4, few_shot_examples["stage_2_few_shot_examples"])
    if generated_text is None:
        return is_relevant
    
    print(f"Stage 2:\n{generated_text}\n\n") 

    if year_guess is not None:
        addition = f"\nIn case this is helpful, the data that your input comes from was published in {year_guess}"
        generated_text = get_chatgpt_response(system_messages["system_message_stage_3"] + addition, generated_text, 0, use_gpt4=False, examples=few_shot_examples["stage_3_few_shot_examples"])
    else:
        generated_text = get_chatgpt_response(system_messages["system_message_stage_3"], generated_text, 0, use_gpt4=False, examples=few_shot_examples["stage_3_few_shot_examples"])

    print(f"Stage 3: \n{generated_text}")
    if generated_text is None:
        return is_relevant

    parsed_response, state_cache = parse_response(generated_text, outbreak_df, system_messages["system_message_stage_4"], general_state=state, state_cache=state_cache, publish_year=year_guess)
    print(f"Parsed response:\n{parsed_response}")
    if parsed_response is not None:
        outbreak_df = parsed_response
    else:
        print("Error: Could not parse the response.")

    # if there was data to be found, add it to dataframe list
    if not outbreak_df.empty:
        outbreak_df['File Name'] = os.path.basename(file)
        outbreak_df['Study'] = outbreak_df['File Name'].map(study_indices)
        print(found_valid_sources)
        if len(found_valid_sources) > 0:
            outbreak_df['Source'] = ' | '.join(found_valid_sources)
        else:
            outbreak_df['Source'] = 'No identified sources'
        print(outbreak_df.iloc[0]['Source'])

        # Append the dataframe to the list
        data_list.append(outbreak_df)

    return is_relevant

def check_for_multiple_locations(location):
    # used to determine if a particular piece of location data refers to more than one location
    # if it does, than the data is probably not accurate
    # example: 1938-1980,"manitoba, ontario, quebec, new brunswick, nova scotia, prince edward island, newfoundland, maine, minnesota",49.8955367,-97.1384584,1,No identified sources,Hardy et al. 1986 atlas of outbreaks.pdf,
    # ^ I doubt that there was a synchronized outbreak across half of canada. This data is bad :/

    response = get_chatgpt_response(system_messages["system_message_check_multiple_locations"], location, 0, False, few_shot_examples["multiple_locations_few_shot_examples"])
    if response is None:
        return False
    response = response.lower()

    return yes_or_no(response)

def windows_to_mac_path(windows_path):
    # Check if the input path contains the specified Windows prefix
    if 'E:\\NIMBioS\\SBW\\SBW Literature\\' not in windows_path:
        return windows_path
    
    # Replace the Windows prefix with the Mac prefix
    mac_path = windows_path.replace('E:\\NIMBioS\\SBW\\SBW Literature\\', '/Volumes/ESD-USB/SBW Literature/')
    
    # Replace backslashes with slashes
    mac_path = mac_path.replace('\\', '/')
    
    return mac_path

def mac_to_windows_path(mac_path):
    # Check if the input path contains the specified Mac prefix
    if '/Volumes/ESD-USB/SBW Literature/' not in mac_path:
        return mac_path
    
    # Replace the Mac prefix with the Windows prefix
    windows_path = mac_path.replace('/Volumes/ESD-USB/SBW Literature/', 'E:\\NIMBioS\\SBW\\SBW Literature\\')
    
    # Replace slashes with backslashes
    windows_path = windows_path.replace('/', '\\')
    
    return windows_path

def get_os():
    current_os = platform.system()
    return current_os

def get_n_pdfs(to_parse_df, num_files):
    """
    Returns a list of up to 'num_files' unprocessed PDF file names from a given DataFrame.

    Arguments:
    to_parse_df : pandas.DataFrame
        The DataFrame must contain the following columns:
            - 'file_name': Indicates the name or path of the file.
            - 'been_processed': Indicates if the file has been processed (1) or not (0).
        The 'file_name' may contain either Windows or macOS file paths depending on the underlying OS.
        
    num_files : int
        The number of unprocessed files to return. If there are fewer unprocessed files than 'num_files', 
        all available unprocessed files will be returned.

    Requirements:
    - The DataFrame should not be empty.
    - 'file_name' and 'been_processed' must exist as columns.
    - 'been_processed' should only contain binary values (0 or 1).

    Returns:
    list
        A list containing the file names (or paths) of up to 'num_files' unprocessed PDFs.
    """
    
    # Specify the name of the CSV file
    current_os = get_os()

    if current_os == "Darwin":
        to_parse_df['file_name'] = to_parse_df['file_name'].apply(windows_to_mac_path)
    elif current_os == "Windows":
        to_parse_df['file_name'] = to_parse_df['file_name'].apply(mac_to_windows_path)

    # Filter the rows where 'been_processed' is equal to 0
    filtered_df = to_parse_df[to_parse_df['been_processed'] == 0]

    # Get up to the first 'num_files' 'file_name' entries from the filtered DataFrame
    # and put them into a list
    file_names_to_process = filtered_df['file_name'].head(num_files).tolist()

    return file_names_to_process

def list_all_pdfs(folder_path):
    # Use os.path.join and '**' to search within the folder and all its subdirectories.
    # '*.pdf' matches all pdf files.
    search_path = os.path.join(folder_path, '**', '*.pdf')
    
    # Use glob to find all the pdf files matching the search_path
    pdf_files = glob.glob(search_path, recursive=True)
    
    return pdf_files

#_________________________________________________________________________


# set system_messages for each stage
system_messages = {
    "system_message_topic_checker": "You are a yes-or-no machine. This means that you only output 'yes' or 'no', and nothing else. You will be given an excerpt from a text, and you will determine if it includes any information about Eastern Spruce Budworms, sometimes written as just Spruce Budworm, or SBW. Only Eastern spruce budworm counts, NOT WESTERN. Say 'yes' if it includes information and 'no' if it does not.",

    "system_message_stage_0": "You are a list-maker making a comma-separated list of sources for research papers about spruce budworms. You are given an excerpt from the text and must determine where the data is coming from. Your possible list items are: Dendrochronological samples from tree cores, Dendrochronological samples from historical buildings, Pheromone traps, Aerial defoliation survey, Survey from insect laboratory, or Personal Communication with the Department of Lands and Forest representative. If the paper uses multiple sources, list each one separately, using commas as delimiters. If no information about the methods of data collection are given, simple output 'Unknown'. It is of the utmost importance that your output is a comma-separated list. Do not write headers or any additional information. Preface the information with 'Data collection method: '.",

    "system_message_stage_0b": "You are a scientist that is extracting the location of study sites from research papers about Spruce Budworm (SBW) outbreaks. You are given an excerpt from a text and must determine if the paper gives exact geographic coordinates of the study sites. You must output the geographic coordinates exactly how it is written, and nothing else. If you don't find the coordinates, output 'Unknown.' The coordinates MUST be numeric. Preface the information with 'Location: '. You must be concise because your output will be parsed as coordinates.",

    "system_message_stage_0c": "You are a classification engine that determines the format of geocoordinate data for researchers. You are given a coordinate pulled from a research paper and you must guess whether it is a bounding box, an individual coordinate in degrees/minutes/seconds, an individual point in decimal degrees, or an invalid/incomplete location (i.e. it is not a two-dimensional bounding box, a single location, or it is not numeric geocoordinates, just a place). Valid coordinates must include latitude and longitude. Your options are 'bounding box', 'degrees/minutes', 'degrees/minutes/seconds', 'decimal degrees', and 'invalid'. Your output must be one of these options and NOTHING ELSE.",

    "system_message_stage_0c_boundingbox": "You are a formatting machine that takes unformatted bounding box coordinates and puts them into a standardized format. You will put all bounding boxes into this format: degree1°N-degree2°N, degree1°W-degree2°W. Each degree may be just a degree, or a decimal degree, a coordinate in degrees/minutes/seconds, etc. Just output this data in the right format. Your output must be this format and NOTHING ELSE.",

    "system_message_stage_0c_dm": "You are a formatting machine that takes unformatted coordinates in degrees/minutes and puts them into a standardized format. You will put all bounding boxes into this format: degree1°minute1'N, degree2°minute2'W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.",

    "system_message_stage_0c_dms": "You are a formatting machine that takes unformatted coordinates in degrees/minutes/seconds and puts them into a standardized format. You will put all bounding boxes into this format: degree1°minute1'second1\"N, degree2°minute2'second2\"W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.",

    "system_message_stage_0c_dd": "You are a formatting machine that takes unformatted coordinates in decimal degrees and puts them into a standardized format. You will put all bounding boxes into this format: degree1.decimal1°N, degree2.decimal2°W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.",

    "system_message_stage_0d": "You are a scientist that is extracting the location of study sites from research papers about Spruce Budworm (SBW) outbreaks. You are given an excerpt from a text and must determine where the study site is located. The location you output must encompass the entire study area and must be locatable on a map using the GeoPy geoservice. If the text gives exact coordinates, output those coordinates exactly and stop. Otherwise, output the location in the following format: Province/State/Municipality, Country. If the study area takes place in the northern/southern/western/eastern part or a specific lake/town/landmark in the municipality/province/state, be sure to include that info. If there is not data about the study area or you only know the country or continent it takes place in, simply print 'Unknown'. Preface the information with 'Location: '. Be concise because your output will be parsed as csv data.",

    "system_message_year_guesser": 'You are a text analysis machine that is inferring the dates from texts. Please output the year that you think it had most likely been published in. It is of the utmost importance that you only output the individual year, and nothing else. No punctuation.',

    "system_message_stage_1": "You are a scientist extracting data from research papers about Spruce Budworm (SBW) infestations and outbreaks. You are to log every instance in which the text refers to a Spruce Budworm outbreak during any years and region. You must only include the SPECIFIC ranges of years and the SPECIFIC region of the data. The region must be locatable on a map. Be as specific as possible. General locations like 'study site' or 'tree stand #3' are not relevant. Include outbreaks whose existence is uncertain. Never include research citations from the text. Only report information related to specific SBW outbreaks in specific years and locations. ALL INFORMATION MUST BE ABOUT SPRUCE BUDWORMS",

    "system_message_stage_2": "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbrea/infestation occured during a specific year or range of years and at a specific geographic location.\n\nFor each instance, output should be a new line in this format, with no headers or labels included.\n\nThe geographic location must be identifiable on a map and can be a city, county, specific lake, etc. Do not include nonspecific or nonidentifiable locations like 'study site'.\n\nIf an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'.\n\nIf the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'. If there was a Spruce Budworm management or control operation in a region, that means that there was a Spruce Budworm outbreak there.\n\nIt is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks.\n",

    "system_message_stage_3": "You are a formatting machine that takes output from ChatGPT and ensures that it is formatted correctly. Each line must be in csv format as follows:\n\"Location\", \"Year(s)\", \"Outbreak (Always only a 'yes', 'no' or 'uncertain')\".\nEach feature must be enclosed by double quotes.\nIf the outbreak lasts multiple years, list the first year, then the last year with a '-' between them (first year-last year). Only list specific years and ranges.\nIf the last feature instead says there was any amount of defoliation or tree mortality, that is also considered a 'yes' for the outbreak feature.\nIt is of the utmost importance that you print only the one piece of data, and absolutely nothing else. Do not say anything other than returning the formatted line. Do not respond like a human. I need perfect CSV text format out of ChatGPT.",

    "system_message_check_multiple_locations": "You are a yes-or-no machine. This means that you only output 'yes' or 'no', and nothing else. You are given a string and you must determine if the string represents locations from more than one state/province. Answer with 'yes' if more than one state/province is represented and 'no' if only one state/province is represented. Sometimes locations are succeeded by the province and country they inhabit, like 'lac seul area, northwestern ontario, canada', and it is still located in only one province. It is of the utmost importance that you print 'yes' or 'no' and nothing else. Do not say anything other than returning the answer. Do not respond like a human.",

    "system_message_stage_4": "You are a computer made to give scientists town names within an area. You will be given a location in North America. Your task is to give a town that belongs at that location to be used as a locality string for GEOLocate software. If the area is very remote, give the nearest town. Put it in csv format as the following: \"city, state, country\". It is of the utmost importance that you print only the one piece of data, and absolutely nothing else. You must output a city name, even if the given area is very large or very remote."
}

few_shot_examples = {
    "stage_1_few_shot_examples": [
        {"role": "user", "content": "We reconstructed the SBW outbreak history at the northern limit of the temperate forest in southern Quebec using dendrochronological material from old buildings and five old-growth stands. Nine potential outbreaks were identified (1976–1991, 1946–1959, 1915–1929, 1872–1903, 1807–1817, 1754–1765, 1706–1717, 1664–1670, and 1630–1638) with three additional uncertain outbreaks (1647–1661, 1606–1619, and 1564–1578)."},
        {"role": "assistant", "content": "-Outbreak: 1976-1991\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1946-1959\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1915-1929\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1872-1903\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1807-1817\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1754-1765\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1706-1717\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1664-1670\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1630-1638\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1647-1661 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1606-1619 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1564-1578 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec"},
        {"role": "user", "content": "Earlier potential outbreaks were identified using dendro-\nchronological material from old buildings only. Three addi-\ntional and distinct growth reductions occurred between the\nbeginning of the 18th and the early 19th centuries in 1807–\n1817 (O5), 1754–1765 (O6), and 1706–1717 (O7) (Fig. 2). (Berguet 1954)"},
        {"role": "assistant", "content": "No specific instances of SBW outbreaks with geographic locations and years were mentioned in the text."},
        {"role": "user", "content": "Several points\n in the neighbourhood of Sioux Lookout and the\n eastern portion of Lac Seul show 1866 as the\n first year of suppression, indicating that the in-\n festation had its origin in this vicinity. Within\n this area, the first apparent year of suppression\n was as early as 1862 in two localities. However,\n in both these localities the white spruce trees\n sampled were very old and showed poor diameter\n growth for approximately the last 100 years, thus\nobscuring to some extent the initiation of suppres-\n sion caused by the spruce budworm (see data for\n Minnitaki Lake, Fig. 3). From this area of\n origin the infestation spread mostly westward un-\n til by 1870 and 1871 it reached points close to\n the Manitoba boundary."},
        {"role": "assistant", "content": "1. Region: Sioux Lookout and the eastern portion of Lac Seul\n  Years: 1862-1871\n"},
        {"role": "user", "content": "Spruce beetle outbreaks have been re ported in the Rocky Mountains of western North America since the late 1800s. In their classic paper, Spruce Beetle in the Rockies,  Schmid and Frye reviewed the literature that emerged from the extensive outbreaks in Colorado in the 1940s."},
        {"role": "assistant", "content": "No instances of Spruce Budworm outbreaks recorded in the text."}
    ],
    "stage_2_few_shot_examples": [
        {"role": "user", "content": "Specific SBW Outbreaks:\n\n1. Outbreak: O1 (1905-1930)\n-Region: Eastern Quebec, east of the St. Lawrence River\n\n2. Outbreak: O2 (1935-1965)\n  -Region: Northern reach of southern Quebec\n\n3. Outbreak: O3 (1968-1988)\n-Region: Southwestern Quebec, along a southwest-northeast transect across central southern Quebec"},
        {"role": "assistant", "content": 'Stage 2:\n"Eastern Quebec, east of the St. Lawrence River", "1905-1930", "Yes"\n"Northern reach of southern Quebec", "1935-1965", "Yes"\n"Southwestern Quebec, along a southwest-northeast transect across central southern Quebec", "1968-1988", "Yes"'},
        {"role": "user", "content": "Stage 1:\n-Outbreak: 1976-1991\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1946-1959\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1915-1929\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1872-1903\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1807-1817\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1754-1765\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1706-1717\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1664-1670\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1630-1638\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1647-1661 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1606-1619 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1564-1578 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec"},
        {"role": "assistant", "content": '"Northern limit of the temperate forest in southern Quebec", "1976-1991", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1946-1959", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1915-1929", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1872-1903", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1807-1817", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1754-1765", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1706-1717", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1664-1670", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1630-1638", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1647-1661", "Uncertain"\n"Northern limit of the temperate forest in southern Quebec", "1606-1619", "Uncertain"\n"Northern limit of the temperate forest in southern Quebec", "1564-1578", "Uncertain"'},
        {"role": "user", "content": "Specific SBW outbreaks in specific years and locations mentioned in the text are as follows:\n\n1. Outbreak in 1989: Two epicenters were identified during this outbreak.\n2. Outbreak in 1991: Four epicenters were identified during this outbreak.\n3. Outbreak in 1996: One epicenter was identified during this outbreak.\n4. Outbreak in 2006: Four epicenters were identified during this outbreak.\n5. Outbreak in 2011: One epicenter was identified during this outbreak.\n\nPlease note that the specific locations of these epicenters were not provided in the text."},
        {"role": "assistant", "content": "No specific instances of SBW outbreaks with geographic locations and years were mentioned in the text."},
        {"role": "user", "content": "-Spruce beetle outbreaks have been reported in the Rocky Mountains of western North America since the late 1800s.\n-A new wave of outbreaks has affected Rocky Mountain subalpine spruce-fir forests beginning in the mid-1980s and continuing to the present.\n-Recent spruce beetle outbreaks in Utah alone have resulted in the deaths of over 1 million spruce trees and the loss of over 90% of the large spruce component in affected stands.\n-Just over 500,000 ha of spruce forests have been affected in Colorado and Wyoming since 1996.\n-The frequency of outbreak initiation dates varied, on average, outbreaks within a given geographic area were first reported once every 19 years.\n-Four periods of broad-scale spruce beetle outbreaks occurring from 1843 to 1860, 1882 to 1889, 1931 to 1957, and 2004 to 2010 were evident in the northwestern Colorado chronology.\n-The unprecedented scale of the current outbreaks suggests that warming temperatures attributed to climate change and the region-wide susceptibility of spruce in the spruce-fir zone have had a significant impact on outbreak occurrence.\n-The occurrence of spruce beetle outbreaks and other disturbance events in Alaska and Colorado dating back to the 1700s has been determined from chronologies constructed by utilizing tree-ring release patterns, dates of spruce beetle-induced tree mortality, and years since stand establishment."},
        {"role": "assistant", "content": "No specific instances of Spruce Budworm outbreaks with geographic locations and years were mentioned in the text."}
    ],
    "stage_3_few_shot_examples": [
        {"role": "user", "content": "Outbreak: 1946\nRegion: New Brunswick, Canada\n\nOutbreak: Unknown\nRegion: Algoma Forest, Ontario, Canada\n\nOutbreak: 1989\nRegion: Central New Brunswick, Canada\n\nOutbreak: Unknown\nRegion: Northern New Brunswick, Canada\n\nOutbreak: Unknown\nRegion: New Brunswick, Canada and northern New England, USA"},
        {"role": "assistant", "content": '"New Brunswick, Canada", "1946", "yes"'},
        {"role": "user", "content": 'No specific outbreaks of Spruce Budworm (SBW) were mentioned in the text.\n\n\nThe text does not provide enough information to generate a CSV line.'},
        {"role":"assistant", "content": 'No specific outbreaks of Spruce Budworm (SBW) were mentioned in the text.'},
        {"role": "user", "content": '"North American boreal forests", "several years", "Yes"\n"University of Quebec at Chicoutimi, Québec, Canada", "2015", "Yes"\n"University of Quebec at Chicoutimi, Québec, Canada", "2016", "Yes"'},
        {"role": "assistant", "content": '"University of Quebec at Chicoutimi, Québec, Canada", "2015", "Yes"\n"University of Quebec at Chicoutimi, Québec, Canada", "2016", "Yes"'},
        {"role": "user", "content": "\"Sioux Lookout area, Northwestern Ontario, Canada\", \"1858-1860\", \"Uncertain\""},
        {"role": "assistant", "content": "\"Sioux Lookout area, Northwestern Ontario, Canada\", \"1858-1860\", \"Uncertain\""},
        {"role": "user", "content": "\"Sioux Lookout area, Northwestern Ontario, Canada\", \"1945 onwards\", \"Yes\""},
        {"role": "assistant", "content": "\"Sioux Lookout area, Northwestern Ontario, Canada\", \"1945-\", \"Yes\""},
        # {"role": "user", "content": "\"Lac Seul area of Northwestern Ontario, Canada\", \"Late 19th century\", \"Yes\""},
        # {"role": "assistant", "content": "No outbreaks of Spruce Budworm with specific years given in the text."}
        {"role": "user", "content": "Outbreak: 1858-1865\nRegion: Sioux Lookout area, northwestern Ontario\nOutbreak: Uncertain\nRegion: Points close to the Manitoba boundary\nOutbreak: Uncertain\nRegion: International boundary, Lake of the Woods\nOutbreak: Current infestation\nRegion: Sioux Lookout and Wabaskang Lake area, northwestern Ontario\nOutbreak: Current infestation\nRegion: Approximately 10,000 square miles in northwestern Ontario\nOutbreak: Uncertain\nRegion: Clay Lake District\nOutbreak: Uncertain\nRegion: Cedar Lake District\nOutbreak: Uncertain\nRegion: Past outbreak area, northwestern Ontario\nOutbreak: Uncertain\nRegion: Forest stands with balsam fir dominance"},
        {"role": "assistant", "content": "\"Sioux Lookout Area, Northwestern Ontario\", \"1858-1865\""},
        {"role": "user", "content": "Outbreak: 1910-unknown\nRegion: Ottawa River Valley, Quebec, Canada\n\nOutbreak: 1940-unknown\nRegion: Ottawa River Valley, Quebec, Canada\n\nOutbreak: 1967-unknown\nRegion: Ottawa River Valley, Quebec, Canada\n\nOutbreak: 1783 (uncertain)\nRegion: Ottawa River Valley, Quebec, Canada"},
        {"role": "assistant", "content": "\"Ottawa River Valley, Quebec, Canada\", \"1910\", \"yes\"\n\"Ottawa River Valley, Quebec, Canada\", \"1940\", \"yes\"\n\"Ottawa River Valley, Quebec, Canada\", \"1967\", \"yes\"\n\"Ottawa River Valley, Quebec, Canada\", \"1783\", \"uncertain\"\n"},
        {"role": "user", "content": "Chicoutimi Cathedral, Chicoutimi, Quebec, Canada, 18th century: 1710-1716, 1754-1759\nChicoutimi Cathedral, Chicoutimi, Quebec, Canada, 19th century: 1811-1813, 1835-1841, 1868-1878"},
        {"role": "assistant", "content": "\"Chicoutimi Cathedral, Chicoutimi, Quebec, Canada\", \"1710-1716\", \"yes\"\n\"Chicoutimi Cathedral, Chicoutimi, Quebec, Canada\", \"1754-1759\", \"yes\"\n\"Chicoutimi Cathedral, Chicoutimi, Quebec, Canada\", \"1811-1813\", \"yes\"\n\"Chicoutimi Cathedral, Chicoutimi, Quebec, Canada\", \"1835-1841\", \"yes\"\n\"Chicoutimi Cathedral, Chicoutimi, Quebec, Canada\", \"1868-1878\", \"yes\""}
    ],
    "multiple_locations_few_shot_examples": [
        {"role": "user", "content": "northern maine, usa"},
        {"role": "assistant", "content": 'no'},
        {"role": "user", "content": "chicoutimi cathedral, chicoutimi, quebec, canada"},
        {"role": "assistant", "content": 'no'},
        {"role": "user", "content": "lac seul and lake nipigon regions, northwestern ontario, canada"},
        {"role": "assistant", "content": 'no'},
        {"role": "user", "content": "manitoba, ontario, quebec, new brunswick, nova scotia, prince edward island, newfoundland, maine, minnesota"},
        {"role": "assistant", "content": 'yes'},
        {"role": "user", "content": "zone b (quebec, ontario, manitoba)"},
        {"role": "assistant", "content": 'yes'},
        {"role": "user", "content": "laurentide park, quebec"},
        {"role": "assistant", "content": 'no'},    
        {"role": "user", "content": "Saguenay and Quebec"},
        {"role": "assistant", "content": 'no'}
    ]


}

end_message = " END\n\n"

def main():
    global pdf_df

    try:

        # get metadata
        valid_sources, outbreak_occurence_values, use_gpt4, results_folder, user_agent = set_metadata()
        
        # check metadata
        if len(results_folder) <= 0:
            raise ValueError("Results folder must be defined in the set_metadata() function")
        if len(user_agent) <= 0:
            raise ValueError("User_agent variable must be defined in the set_metadata() function")

        file_name = find_unused_filename(results_folder, "results")
        print(file_name)


        # Initialize the count for relevant files
        num_relevant = 0

        # Determine the OS to set the appropriate CSV file path
        current_os = get_os()

        # Where I stored the info about which pdfs have/have not been parsed
        csv_filename = f'{results_folder}/all_pdfs.csv' if current_os == 'Darwin' else f'{results_folder}\\all_pdfs.csv'

        # Load existing CSV data into DataFrame
        pdf_df = pd.read_csv(csv_filename)

        # Fetch a list of PDF files to process
        pdf_files = get_n_pdfs(pdf_df, 500)

        print("Processing all files in this directory. This may take a while!")

        # Iterate over each PDF file
        for file_num, file in enumerate(pdf_files, start=1):
            print(f"File {file_num}/{len(pdf_files)}")

            # Convert the file path based on OS
            windows_file = mac_to_windows_path(file)

            # Process individual file
            relevant = process_file(file)

            # Mark the file as processed in the DataFrame
            pdf_df.loc[pdf_df['file_name'] == windows_file, 'been_processed'] = 1

            # Update DataFrame if the file is relevant
            if relevant:
                num_relevant += 1
                pdf_df.loc[pdf_df['file_name'] == windows_file, 'relevance'] = 1

        # Print summary statistics
        print(f"{num_relevant}/{len(pdf_files)} files were analyzed")

    # Handle keyboard interruption
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt\n")

    # Handle other exceptions
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")

    # Always execute cleanup
    finally:
        end_runtime()


"""
Defining global variables
"""

# for concatenating dataframes
data_list = []

# used to rewrite all_pdfs.csv data (used for account which texts were analyzed, found to be relevant)
pdf_df = pd.DataFrame()

state_cache = {}
location_coordinates = {}

if __name__ == "__main__":
    main()
