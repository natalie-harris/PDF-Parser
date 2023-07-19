"""
Natalie Harris, NIMBioS
5/23/23

This program extracts text from downloaded pdf, feeds it and a system_message to openai api, and retrieves information parsed by chatgpt

Note: https://arxiv.org/pdf/2306.11644.pdf
        ^ this is a cool paper
"""

import os
import time
import socket
import re
import PyPDF2
import openai
import tiktoken
import glob
from geopy.geocoders import Nominatim
from geopy.adapters import AdapterHTTPError
from geopy.exc import GeocoderUnavailable
from geopy.exc import GeocoderServiceError
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic
import pandas as pd

# used for checking how long a text would be in chatgpt tokens
def get_tokenized_length(text, model):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# helper function to get chatgpt output from set of inputs
# note on temperature from OpenAI: "OpenAI models are non-deterministic, meaning that identical inputs can yield different outputs. Setting temperature to 0 will make the outputs mostly deterministic, but a small amount of variability may remain."
# see https://community.openai.com/t/the-system-role-how-it-influences-the-chat-behavior/87353 for suggested ideal system message placement
def get_chatgpt_response(system_message, user_message, temp, use_gpt4=False):
    gpt_model = ''
    total_message = system_message + user_message
    if use_gpt4:
        num_tokens = get_tokenized_length(total_message, 'gpt-4')
        gpt_model = 'gpt-4'
        # if num_tokens < 8192:
        #     gpt_model = 'gpt-4'
        # else:
        #     gpt_model = 'gpt-4-32k'
    else:
        num_tokens = get_tokenized_length(total_message, 'gpt-3.5-turbo')
        if num_tokens < 4096:
            gpt_model = 'gpt-3.5-turbo'
        else:
            gpt_model = 'gpt-3.5-turbo-16k'

    got_response = False
    while not got_response:
        try:
            response = openai.ChatCompletion.create(
                model = gpt_model,
                messages = [
                    {"role": "user", "content": user_message},
                    {"role": "system", "content": system_message}
                ],
                temperature = temp
            )
            generated_text = response['choices'][0]['message']['content']
            got_response = True
            return generated_text

        except openai.error.RateLimitError as err:
            if 'You exceeded your current quota' in str(err):
                print("You've exceeded your current billing quota. Go check on that!")
                exit()
            num_seconds = 3
            print(f"Waiting {num_seconds} seconds due to high volume of {gpt_model} users.")
            time.sleep(3)

        except openai.error.APIError as err:
            print("An error occured. Retrying request.")

        except openai.error.Timeout as err:
            print("Request timed out. Retrying...")

        except openai.error.ServiceUnavailableError as err:
            num_seconds = 3
            print(f"Server overloaded. Waiting {num_seconds} seconds and retrying request.")

def get_latitude_longitude(town, state, country):
    geolocater = Nominatim(user_agent = "research_paper_parser mzg857@vols.utk.edu")
    has_coords = False
    retries = 0
    while not has_coords:
        try:
            time.sleep(1)
            location = geolocater.geocode(query=f"{town}, {state}, {country}")
            has_coords = True
        except (AdapterHTTPError, socket.timeout) as e:
            retries += 1
            if retries > 5:
                print(f"Attempted {retries} retries. Moving on...")
                return None, None
            print("There was an HTTP error getting coordinates. Retrying...")
        except GeocoderUnavailable as e:
            retries += 1
            if retries > 5:
                print(f"Attempted {retries} retries. Moving on...")
                return None, None
            print("Geopy error. Waiting 5 seconds.")
            time.sleep(5)
        except GeocoderServiceError as e:
            print("Geopy error. Moving on...")
            return None, None
            
    if location is None:
        return None, None # must handle when location is not found

    latitude = location.latitude
    longitude = location.longitude

    return latitude, longitude

def location_to_coordinates(location, system_message):
    user_message = 'The boreal forest in ' + location 
    temperature = 0

    print("location to coords")

    generated_text = get_chatgpt_response(system_message, user_message, temperature)
    print(f"More specific location: {generated_text}")
    generated_text = generated_text.lower().strip().split(',')
    if len(generated_text) != 3:
        return None, None
    city = generated_text[0]
    state = generated_text[1]
    country = generated_text[2]

    latitude, longitude = get_latitude_longitude(city, state, country)
    return latitude, longitude

# helper function to remove commas from location values
def make_csv_format(line):
    split_line = line.lower().strip().split(',')
    length = len(split_line)

    if length < 5:
        return line
    
    line = ""

    line += split_line[0]
    for i in range(1, length - 3):
        line += ' '
        line += split_line[i].strip()

    line += (", " + split_line[length - 3])
    line += (", " + split_line[length - 2])
    line += (", " + split_line[length - 1])

    return line

# returns a list of outbreak data at a location for each year if it is delivered as a range of years
def list_each_year(original_line):
    print(f"original line: {original_line} Got here!")
    split_line = original_line.split(',')
    location = split_line[0].strip()
    years = split_line[1].strip()
    outbreak = split_line[2].strip()
    first_year = years[:4]
    last_year = years[-4:]

    # error handling
    if not first_year.isdigit() or not last_year.isdigit():
        return [original_line]
    
    first_year = int(first_year)
    last_year = int(last_year)
    if first_year >= last_year or last_year - first_year > 50 or first_year > 2022 or last_year > 2023: #filtering out invalid year values and when difference between two years > 50
        return [original_line]
    
    new_list = []
    for i in range(first_year, last_year + 1):
        strings = [location, str(i), outbreak]
        new_line = ", ".join(strings)
        new_list.append(new_line)

    print(new_list)

    return new_list

# parse gpt output to convert location to lat/long and reduce output clutter
def parse_response(response, outbreak_df, system_message_stage_3, general_latitude=0.0, general_longitude=0.0):
    split_response = response.splitlines()
    new_split_response = [] # used for adding additional lines when chatgpt gives range of years

    # cache previously found location coordinates to reduce chatgpt use
    cached_location_coords = {}

    # for each line in chatgpt's response
    for line in split_response:

        # make sure line is in the correct format, otherwise move to next line
        line = make_csv_format(line)
        split_line = line.split(',')

        if len(split_line) != 3:
            continue

        location = split_line[0].strip().lower().strip('"')
        year = split_line[1].strip().lower().strip('"')
        outbreak = split_line[2].strip().lower().strip('"')

        print(f"Slightly more formatted: {line}\nLocation: {location}, Year: {year}, Outbreak: {outbreak}")

        # print(outbreak != 'yes' and outbreak != 'no' and outbreak != 'uncertain')
        # print(any(char.isalpha() for char in year))
        # print(len(year) != 4 and len(year) != 9)
        # print(len(location) <= 3)
        print("\n")

        if outbreak != 'yes' and outbreak != 'no' and outbreak != 'uncertain':
            continue
        if any(char.isalpha() for char in year):
            continue
        if len(year) != 4 and len(year) != 9:
            continue
        if len(location) <= 3:
            continue

        print("Got here")

        new_line = ", ".join([location, year, outbreak])

        # if data given as range of years, add every year to new list
        print(year)
        if len(year) == 9 and year[4] == '-':
            every_year = list_each_year(new_line)
            print(every_year[0])
            if len(every_year) > 1:
                print(every_year)
                for single_year in every_year:
                    new_split_response.append(single_year)
        else:
            new_split_response.append(new_line)

    for line in new_split_response:

        split_line = line.split(',')
        print(split_line)
        location = split_line[0].lower().strip()
        year = split_line[1].lower().strip()
        outbreak = split_line[2].lower().strip()

        # search for location in cache, otherwise get coordinates and store in cache
        if location in cached_location_coords:
            latitude = cached_location_coords[location][0]
            longitude = cached_location_coords[location][1]
        else:
            latitude, longitude = location_to_coordinates(location, system_message_stage_3)
            print(latitude, longitude)
            if latitude == None or longitude == None:

                # need to implement replacing with coordinates if possible
                if general_latitude != 0.0 and general_longitude != 0.0:
                    latitude = general_latitude
                    longitude = general_longitude
                else:
                    return outbreak_df
                
            cached_location_coords[location] = (latitude, longitude)
        
        # add latitude and longitude to dataframe
        split_line.append(latitude)
        split_line.append(longitude)
        print(f"{location}, {latitude}, {longitude}, {year}, {outbreak}")
        # print(outbreak_df)
        outbreak_df.loc[len(outbreak_df)] = [location, latitude, longitude, year, outbreak, '']



    return outbreak_df

# returns chunk group of required lengths so as to stay under openai token limit
def build_chunk_group(system_message, text, end_message, use_gpt4=False):
    system_message_length = len(system_message) + len(end_message)
    max_token_length = 16000
    if use_gpt4:
        max_token_length = 8000
    base_multiplier = 4
    safety_multiplier = .8 # used just in case local tokenizer works differently than openai's
    chunk_group = []

    i = 0
    while i < len(text):

        # get initial text length values
        multiplier = base_multiplier
        user_message_length = int(max_token_length * multiplier) - system_message_length
        message = system_message + text[i:i+user_message_length] + end_message
        token_length = get_tokenized_length(message, 'gpt-3.5-turbo')
        
        # while text is too long for openai model, keep reducing size and try again
        while token_length > int(max_token_length * safety_multiplier):
            multiplier *= .95
            user_message_length = int(max_token_length * multiplier) - system_message_length
            message = system_message + text[i:i+user_message_length] + end_message
            token_length = get_tokenized_length(message, 'gpt-3.5-turbo')
        
        # add chunk to chunk set, move on
        chunk_group.append([system_message, text[i:i+user_message_length] + end_message])
        i += user_message_length

    return chunk_group

# returns a df with zeros put in between correct values
# also fixes conflicts where two entries with the same coordinates, and year have different outbreak values
def build_dataframe(df):
    


    return df

def dms_to_dd(dms):
    # Check if input is in decimal degree format
    if re.match(r"[-+]?[0-9]*\.?[0-9]+°[NSWE]", dms):
        degree, direction = re.match(r"([-+]?[0-9]*\.?[0-9]+)°([NSWE])", dms).groups()
        dd = float(degree)
        if direction in 'SW':
            dd *= -1
        return dd
    # If not, assume it's in DMS format
    else:
        degrees, minutes, seconds, direction = re.match(r"(\d+)°(\d+)?'?(?:([0-9.]+)?\"?)?([NSWE])", dms).groups()
        dd = float(degrees) + (float(minutes) if minutes else 0)/60 + (float(seconds) if seconds else 0)/3600
        if direction in 'SW':
            dd *= -1
        return dd

def get_centroid_of_bb(bounding_box):
    try:
        print(bounding_box)
        lat1_str, lat2_str, lon1_str, lon2_str = re.match(r"(.+?)-(.+?),\s*(.+?)-(.+)", bounding_box).groups()
        lat1 = dms_to_dd(lat1_str)
        lat2 = dms_to_dd(lat2_str)
        lon1 = dms_to_dd(lon1_str)
        lon2 = dms_to_dd(lon2_str)

        line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)
        midpoint = line.Position(0.5 * line.s13)
        return (midpoint["lat2"], midpoint["lon2"])

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return (None, None)

def parse_coordinates(coordinates):
    try:
        lat_str, lon_str = re.match(r"(.+),\s*(.+)", coordinates).groups()
        lat = dms_to_dd(lat_str)
        lon = dms_to_dd(lon_str)

        return lat, lon
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None

def clean_coordinates(coordinates):
    latitude, longitude = None, None
    temperature = 0

    if not coordinates.startswith('unknown'):

        # print(f"{i+1}/{len(stage0b_chunks)} {generated_text}")            
        contains_numbers = any(char.isdigit() for char in coordinates)

        if not contains_numbers: # just incase there aren't even any numbers in the coordinates
            return latitude, longitude

        coord_classification = get_chatgpt_response(system_message_stage_0c, coordinates, temperature).lower().strip()
        # print(f"Coordinate classification: {coord_classification}")

        if coord_classification in ['bounding box', 'degrees/minutes', 'degrees/minutes/seconds', 'decimal degrees']: # begin coordinate verification
            formatted_coords = 'unknown'
            if coord_classification == 'bounding box':
                formatted_coords = get_chatgpt_response(system_message_stage_0c_boundingbox, coordinates, temperature)
                latitude, longitude = get_centroid_of_bb(formatted_coords)
            elif coord_classification == 'degrees/minutes':
                formatted_coords = get_chatgpt_response(system_message_stage_0c_dm, coordinates, temperature)
                latitude, longitude = parse_coordinates(formatted_coords)
            elif coord_classification == 'degrees/minutes/seconds':
                formatted_coords = get_chatgpt_response(system_message_stage_0c_dms, coordinates, temperature)
                latitude, longitude = parse_coordinates(formatted_coords)
            elif coord_classification == 'decimal degrees':
                formatted_coords = get_chatgpt_response(system_message_stage_0c_dd, coordinates, temperature)
                latitude, longitude = parse_coordinates(formatted_coords)

    return latitude, longitude

def extract_abstract_to_references(text):
    # convert to lowercase for case-insensitive search
    lower_case_text = text.lower()

    try:
        # find the first occurrence of 'abstract'
        start = lower_case_text.index('abstract')
    except ValueError:
        # if 'abstract' is not found, start from the beginning of the text
        start = 0

    try:
        # find the last occurrence of 'references' and adjust to the end of the word
        end = lower_case_text.rindex('references') + len('references')
    except ValueError:
        # if 'references' is not found, end at the last character of the text
        end = len(text)

    # extract the substring from 'abstract' to 'references' (or the beginning/end of the text if either is not found)
    extracted_text = text[start:end]
    return extracted_text

# removes unneccesary tabs, newlines, spaces
def cleanup_text(text):

    to_replace = [[' \t', ' '], [' \n', ' '], [' \'', '\''], ['-   ', '-'], ['-  ', '-'], ['- ', '-'], ['  ', ' '], [' –', '-']]

    for pair in to_replace:
        text = text.replace(pair[0], pair[1])

    return text

def get_study_index(file, study_indices):
    file = file.rsplit('/', 1)[-1]
    if file in study_indices:
        return study_indices[file]
    return None

def get_val_from_dict(key, dict):
    if key in dict:
        return dict[key]
    return None

"""# returns max distance from general 
def get_max_boundary_distance(province, boundary_percent):
    
    return 100

def is_within_boundary_distance(boundary_percent, gen_lat, gen_long, new_lat, new_long):
    distance = geodesic((gen_lat, gen_long), (new_lat, new_long)).km
    print(distance)

    # find out what province/state these coordinates are in
    province = 'Quebec'


    return distance <= get_max_boundary_distance(province, boundary_percent)
"""

#_________________________________________________________________________


# set system_messages for each stage
system_message_stage_0 = "You are a list-maker making a comma-separated list of sources for research papers about spruce budworms. You are given an excerpt from the text and must determine where the data is coming from. Your possible list items are: Dendrochronological samples from tree cores, Dendrochronological samples from historical buildings, Pheromone traps, Aerial defoliation survey, Survey from insect laboratory, or Personal Communication with the Department of Lands and Forest representative. If the paper uses multiple sources, list each one separately, using commas as delimiters. If no information about the methods of data collection are given, simple output 'Unknown'. It is of the utmost importance that your output is a comma-separated list. Do not write headers or any additional information. Preface the information with 'Data collection method: '."

system_message_stage_0b = "You are a scientist that is extracting the location of study sites from research papers about Spruce Budworm (SBW) outbreaks. You are given an excerpt from a text and must determine if the paper gives exact geographic coordinates of the study sites. You must output the geographic coordinates exactly how it is written, and nothing else. If you don't find the coordinates, output 'Unknown.' The coordinates MUST be numeric. Preface the information with 'Location: '. You must be concise because your output will be parsed as coordinates.\n\n"

system_message_stage_0c = "You are a classification engine that determines the format of geocoordinate data for researchers. You are given a coordinate pulled from a research paper and you must guess whether it is a bounding box, an individual coordinate in degrees/minutes/seconds, an individual point in decimal degrees, or an invalid/incomplete location (i.e. it is not a two-dimensional bounding box, a single location, or it is not numeric geocoordinates, just a place). Valid coordinates must include latitude and longitude. Your options are 'bounding box', 'degrees/minutes', 'degrees/minutes/seconds', 'decimal degrees', and 'invalid'. Your output must be one of these options and NOTHING ELSE.\n\n"

system_message_stage_0c_boundingbox = "You are a formatting machine that takes unformatted bounding box coordinates and puts them into a standardized format. You will put all bounding boxes into this format: degree1°N-degree2°N, degree1°W-degree2°W. Each degree may be just a degree, or a decimal degree, a coordinate in degrees/minutes/seconds, etc. Just output this data in the right format. Your output must be this format and NOTHING ELSE.\n\n"

system_message_stage_0c_dm = "You are a formatting machine that takes unformatted coordinates in degrees/minutes and puts them into a standardized format. You will put all bounding boxes into this format: degree1°minute1'N, degree2°minute2'W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.\n\n"

system_message_stage_0c_dms = "You are a formatting machine that takes unformatted coordinates in degrees/minutes/seconds and puts them into a standardized format. You will put all bounding boxes into this format: degree1°minute1'second1\"N, degree2°minute2'second2\"W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.\n\n"

system_message_stage_0c_dd = "You are a formatting machine that takes unformatted coordinates in decimal degrees and puts them into a standardized format. You will put all bounding boxes into this format: degree1.decimal1°N, degree2.decimal2°W. Just output this data in the right format. Your output must be this format and NOTHING ELSE.\n\n"

system_message_stage_0d = "You are a scientist that is extracting the location of study sites from research papers about Spruce Budworm (SBW) outbreaks. You are given an excerpt from a text and must determine where the study site is located. The location you output must encompass the entire study area and must be locatable on a map using the GeoPy geoservice. If the text gives exact coordinates, output those coordinates exactly and stop. Otherwise, output the location in the following format: Province/State/Municipality, Country. If the study area takes place in the northern/southern/western/eastern part or a specific lake/town/landmark in the municipality/province/state, be sure to include that info. If there is not data about the study area or you only know the country or continent it takes place in, simply print 'Unknown'. Preface the information with 'Location: '. Be concise because your output will be parsed as csv data.\n\n"

system_message_stage_1 = "You are a scientist extracting data from research papers about Spruce Budworm (SBW) infestations and outbreaks. You are to log every instance in which the text refers to a Spruce Budworm outbreak during any years and region. You must only include the SPECIFIC ranges of years and the SPECIFIC region of the data. The region must be locatable on a map. Be as specific as possible. General locations like 'study site' or 'tree stand #3' are not relevant. Include outbreaks whose existence is uncertain. Never include research citations from the text. Only report information related to specific SBW outbreaks in specific years and locations."

system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbreak/infestation occured during a specific year or range of years and at a specific geographic location. Write every instance in the following format exactly: The geographic location, then the year, whether there was or was not an outbreak/infestation (always a yes or no), and then a new line. This data must be in csv file format, with commas in between and double quotes around each feature. Never include the header or any labels. The geographic location must be something like a city, a county, a specific lake, or anything that is locatable on a map. If an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'. It is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks. If the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'"

system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbreak/infestation occured during a specific year or range of years and at a specific geographic location. Present your findings in the following consistent format: '\"Geographic location\"', '\"Year or Year range\"', '\"Outbreak presence (Yes/No/Uncertain)\"'. For each instance, output should be a new line in this format, with no headers or labels included. The geographic location, encapsulated within double quotation marks, must be identifiable on a map and can be a city, county, specific lake, etc. It is of the utmost importance that the location must be provincial/state level level or smaller, AKA ONLY INCLUDE locations that are the size of provinces/states or SMALLER. Do not include nonspecific or nonidentifiable locations like 'study site'. If an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'. It is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks. If the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'."

system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbrea/infestation occured during a specific year or range of years and at a specific geographic location.\n\n\
Present your findings in the following consistent format: '\"Geographic location\"', '\"Year or Year range\"', '\"Outbreak presence (Yes/No/Uncertain)\"'.\n\n\
For each instance, output should be a new line in this format, with no headers or labels included.\n\n\
The geographic location must be identifiable on a map and can be a city, county, specific lake, etc. Do not include nonspecific or nonidentifiable locations like 'study site'.\n\n\
If an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'.\n\n\
If the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'.\n\n\
It is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks.\n"

system_message_stage_3 = "You are a computer made to give scientists town names within an area. You will be given a location in North America. Your task is to give a town that belongs at that location to be used as a locality string for GEOLocate software. If the area is very remote, give the nearest town. Put it in csv format as the following: \"city, state, country\". It is of the utmost importance that you print only the one piece of data, and absolutely nothing else. You must output a city name, even if the given area is very large or very remote."

end_message = " END\n\n"

def main():

    # not including 5a, 6a, 8 because they are all Hardy et al.
    # ChatGPT cannot read the pictures in Hardy et al. so we can't compare data
    study_indices = {
        "Bouchard et al. 2018 -1.pdf": 1, 
        "Boulanger et al. 2012 SBW outbreaks 400 yrs.pdf": 2,
        "Fraver et al. 2006 time series SBW Maine.pdf": 3,
        "Navarro et al. 2018 space time SBW.pdf": 4,
        "Elliot 1960.pdf": 5,
        "Blais 1954.pdf": 6,
        "Blais 1981.pdf": 7,
        "Berguet et al. 2021 spatiotemp dyn 20th cent sbw.pdf": 9
        }

    valid_sources = [
        'dendrochronological samples from tree cores', 
        'dendrochronological samples from historical buildings', 
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

    general_locations = {
        1: 'quebec, canada',
        2: 'southern quebec'
    }

    general_coords = {
        1: (46.907330, -71.389520),
        2: (46.505566, -73.347985)
    }

    location_coordinates = {}

    max_boundary_percentage = .5

    file_name = "Testing/testing_data/test17"

    use_gpt4 = True

    # for concatenating dataframes
    data_list = []

    # get folder path and file name of pdf, create pdf reader instance
    pdf_files = glob.glob("papers/*.pdf")
    print("Processing all files in this directory. This may take a while!")
    for file in pdf_files:

        if file != 'papers/Boulanger et al. 2012 SBW outbreaks 400 yrs.pdf' and file != 'papers/Bouchard et al. 2018 -1.pdf':
            continue

        print(f"Currently Processing: {file}")

        # file_name = input('Input the name of the file you would like to parse: ')
        # file = open(file_name, 'rb')
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        # concat all text into pdf_text string
        pdf_text = ''
        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            pdf_text += page.extract_text().replace("-\n", "").replace("\n", " ").replace(" -", "-")

        # remove everything before abstract and after/on references page
        pdf_text = extract_abstract_to_references(pdf_text)
        pdf_text = cleanup_text(pdf_text)

        # set up openai api
        openai_key = "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"
        openai.api_key = openai_key
        model_list = openai.Model.list()

        # # 1. get source of data
        # # build chunks
        # source_chunk_group = build_chunk_group(system_message_stage_0, pdf_text, end_message)
        # source_prefix = "data collection method: "

        # # iterate through each chunk until source is found
        # source = 'unknown'
        # for chunk in source_chunk_group:
        #     system_message = chunk[0]
        #     user_message = chunk[1]
        #     temperature = 0
        #     generated_text = get_chatgpt_response(system_message, user_message, temperature).lower()
        #     if generated_text.startswith(source_prefix):
        #         generated_text = generated_text[len(source_prefix):]
        #     if generated_text.endswith('.'):
        #         generated_text = generated_text[0:len(generated_text) - 1]
        #     if not generated_text.startswith('unknown'):
        #         print(generated_text)
        #         source = generated_text
        #     if source != 'unknown':
        #         break

        # found_valid_sources = []
        # source = source.split(',')
        # for i in range(len(source)):
        #     source[i] = source[i].strip().lower()
        #     if source[i] in valid_sources:
        #         found_valid_sources.append(source[i])
        
        # print(found_valid_sources)

        # 2. get location of data to use in case location cannot be found
        stage0b_chunks = build_chunk_group(system_message_stage_0b, pdf_text, end_message)
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
            while i < len(stage0b_chunks) and location == 'unknown':
                chunk = stage0b_chunks[i]
                system_message = chunk[0]
                user_message = chunk[1]
                temperature = 0

                generated_text = get_chatgpt_response(system_message, user_message, temperature).lower()

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


        # 2.5. try to find the general area of the study and get coordinates from geopy if coordinates are not already found
        if gen_location is None:

            stage0c_chunks = build_chunk_group(system_message_stage_0d, pdf_text, end_message, use_gpt4)
            location_prefix = 'location: '

            location = 'unknown'
            i = 0
            while i < len(stage0c_chunks) and location == 'unknown':
                chunk = stage0c_chunks[i]
                system_message = chunk[0]
                user_message = chunk[1]
                temperature = 0

                generated_text = get_chatgpt_response(system_message, user_message, temperature, use_gpt4).lower()
                print(f'Generated General Location: {generated_text}')

                if generated_text.startswith(location_prefix):
                    generated_text = generated_text[len(location_prefix):]
                if generated_text.endswith('.'):
                    generated_text = generated_text[0:len(generated_text) - 1]
                if 'unknown' not in generated_text:
                    location = f'"{generated_text}"'
                    if latitude is None or longitude is None:
                        if location not in location_coordinates:
                            latitude, longitude = location_to_coordinates(location, system_message_stage_3)
                            if latitude is not None and longitude is not None:
                                location_coordinates[location] = (latitude, longitude)
                                found_coordinates = True
                        else:
                            latitude, longitude = location_coordinates[location][0], location_coordinates[location][1]
                        latitude, longitude = location_to_coordinates(location, system_message_stage_3)
                        if latitude is not None and longitude is not None:
                            found_coordinates = True

                i += 1

        else:
            location = gen_location

        if found_coordinates:
            print(latitude, longitude)
        print(location)


        # if not found_coordinates, then we need to do the chatgpt process to get coordinates from geopy

        # set up dataframe for csv output
        outbreak_df = pd.DataFrame(columns=['area', 'Latitude', 'Longitude', 'Year', 'Outbreak', 'Source'])

        # build prompt chunks
        chunk_group = build_chunk_group(system_message_stage_1, pdf_text, end_message, use_gpt4)

        stage1_results = ''

        # make api call for each chunk in each chunk_group, print response
        for chunk in chunk_group:
            system_message = chunk[0]
            user_message = chunk[1]
            temperature = 0

            print(f'Text chunk: {user_message}')

            generated_text = get_chatgpt_response(system_message, user_message, temperature, use_gpt4)
            generated_text = cleanup_text(generated_text)
            stage1_results += f'\n{generated_text}'

        print(f"\nStage 1: {stage1_results}\n\n")

    #     generated_text = get_chatgpt_response(system_message_stage_2, stage1_results, 0, use_gpt4)
    #     print(f"Stage 2:\n{generated_text}\n\n")    
        
    #     parsed_response = parse_response(generated_text, outbreak_df, system_message_stage_3)
    #     if parsed_response is not None:
    #         outbreak_df = parsed_response
    #     else:
    #         print("Error: Could not parse the response.")

    #     # if there was data to be found, add it to dataframe list
    #     if not outbreak_df.empty:
    #         outbreak_df['File Name'] = os.path.basename(file)
    #         outbreak_df['Study'] = outbreak_df['File Name'].map(study_indices)
    #         print(found_valid_sources)
    #         if len(found_valid_sources) > 0:
    #             outbreak_df['Source'] = ' | '.join(found_valid_sources)
    #         else:
    #             outbreak_df['Source'] = 'No identified sources'
    #         print(outbreak_df.iloc[0]['Source'])

    #         # Append the dataframe to the list
    #         data_list.append(outbreak_df)

    #         # Create individual csv file for this study
    #         # file_name_no_extension = os.path.splitext(file)[0]
    #         # csv_file_name = 'outbreak_data_' + file_name_no_extension + '.csv'
    #         # excel_file_name = 'outbreak_data_' + file_name_no_extension + '.xlsx'
    #         # outbreak_df.to_csv(csv_file_name, index=False)
    #         # outbreak_df.to_excel(excel_file_name, index=False)

    #     print(outbreak_df)

    # # concatenate all dataframes
    # final_data = []
    # for df in data_list:

    #     data = df
    #     filename = data['File Name'].iloc[0]
    #     source = data['Source'].iloc[0]
    #     study = data['Study'].iloc[0]
    #     list_data_filled = []
    #     data = data.sort_values(['area', 'Year'])
        
    #     for area in data['area'].unique():
    #         area_data = data[data['area'] == area].copy()
            
    #         # Convert 'Year' column to int
    #         area_data['Year'] = area_data['Year'].astype(int)
            
    #         min_year = int(area_data['Year'].min())
    #         max_year = int(area_data['Year'].max())
    #         latitude = area_data['Latitude'].iloc[0]
    #         longitude = area_data['Longitude'].iloc[0]
            
    #         all_years = pd.DataFrame({'Year': range(min_year - 1, max_year + 2)})
    #         all_years['area'] = area
    #         all_years['Latitude'] = latitude
    #         all_years['Longitude'] = longitude
            
    #         # Convert 'area', 'Latitude', and 'Longitude' in both DataFrames to the same data type if needed
    #         # e.g., all_years['area'] = all_years['area'].astype(str)
    #         # area_data['area'] = area_data['area'].astype(str)
            
    #         merged_data = pd.merge(all_years, area_data, how='left', on=['Year', 'area', 'Latitude', 'Longitude'])
    #         merged_data['Outbreak'].fillna('no', inplace=True)
    #         merged_data['Study'] = study
    #         merged_data['File Name'] = filename
    #         merged_data['Source'] = source
    #         list_data_filled.append(merged_data)

    #     data = pd.concat(list_data_filled, ignore_index=True)

    #     data['Outbreak'] = data['Outbreak'].map(outbreak_occurence_values)
    #     final_data.append(data)

    # if len(final_data) > 0:
    #     all_data = pd.concat(final_data, ignore_index=True)
    #     all_data.to_csv(file_name + '.csv', index=False)
    #     all_data.to_excel(file_name + '.xlsx', index=False)

if __name__ == "__main__":
    main()
