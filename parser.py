"""
Natalie Harris, NIMBioS
5/23/23

This program extracts text from downloaded pdf, feeds it and a system_message to openai api, and retrieves information parsed by chatgpt
"""

import os
import time
import PyPDF2
import openai
import tiktoken
import glob
from geopy.geocoders import Nominatim
import pandas as pd




# helper function to get chatgpt output from set of inputs
def get_chatgpt_response(system_message, user_message, temp):
    gpt_model = 'gpt-3.5-turbo'

    got_response = False
    while not got_response:
        try:
            response = openai.ChatCompletion.create(
                model = gpt_model,
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
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

def get_latitude_longitude(town, state, country):
    geolocater = Nominatim(user_agent = "research_paper_parser")
    has_coords = False
    while not has_coords:
        try:
            location = geolocater.geocode(query=f"{town}, {state}, {country}")
            has_coords = True
        except geopy.adapters.AdapterHTTPError as err:
            print("There was an HTTP error getting coordinates. Retrying...")
    
    if location is None:
        return None, None # must handle when location is not found

    latitude = location.latitude
    longitude = location.longitude

    return latitude, longitude

def location_to_coordinates(location):
    system_message = "You are a computer made to give scientists town names \
                        within an area. You will be given a location in North America. Your \
                        task is to give a town that belongs near or at that location to \
                        be used as a locality string for GEOLocate software. \
                        Put it in csv format as the following: \"city, \
                        state, country\". It is of the utmost importance that you \
                        print only the one piece of data, and absolutely nothing else."
    user_message = location 
    temperature = .3

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

    if length < 4:
        return line
    
    line = ""

    line += split_line[0]
    for i in range(1, length - 2):
        line += ' '
        line += split_line[i].strip()

    line += (", " + split_line[length - 2])
    line += (", " + split_line[length - 1])

    return line

# returns a list of outbreak data at a location for each year if it is delivered as a range of years
def list_each_year(original_line):
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
    if first_year >= last_year:
        return [original_line]
    
    new_list = []
    for i in range(first_year, last_year + 1):
        strings = [location, str(i), outbreak]
        new_line = ", ".join(strings)
        new_list.append(new_line)

    return new_list


# parse gpt output to convert location to lat/long and reduce output clutter
def parse_response(response, outbreak_df):
    split_response = response.splitlines()
    new_split_response = [] # used for adding additional lines when chatgpt gives range of years

    # cache previously found location coordinates to reduce chatgpt use
    cached_location_coords = {}

    # for each line in chatgpt's response
    for line in split_response:
        print(line)

        # make sure line is in the correct format, otherwise move to next line
        line = make_csv_format(line)
        #print(f"Slightly more formatted: {line}")
        split_line = line.split(',')

        if len(split_line) != 3:
            continue

        location = split_line[0].lower().strip()
        year = split_line[1].lower().strip()
        outbreak = split_line[2].lower().strip()

        if outbreak != 'yes' and outbreak != 'no':
            continue
        if any(char.isalpha() for char in year):
            continue
        if len(year) != 4 and len(year) != 9:
            continue
        if len(location) <= 3:
            continue

        # if data given as range of years, add every year to new list
        if len(year) == 9 and year[4] == '-':
            every_year = list_each_year(line)
            print(every_year)
            for single_year in every_year:
                new_split_response.append(single_year)

    for line in new_split_response:

        split_line = line.split(',')
        location = split_line[0].lower().strip()
        year = split_line[1].lower().strip()
        outbreak = split_line[2].lower().strip()

        # search for location in cache, otherwise get coordinates and store in cache
        if location in cached_location_coords:
            latitude = cached_location_coords[location][0]
            longitude = cached_location_coords[location][1]
        else:
            latitude, longitude = location_to_coordinates(location)
            if latitude == None or longitude == None:
                return outbreak_df
            cached_location_coords[location] = (latitude, longitude)
        
        # add latitude and longitude to dataframe
        split_line.append(latitude)
        split_line.append(longitude)
        print(f"{location}, {latitude}, {longitude}, {year}, {outbreak}")
        outbreak_df.loc[len(outbreak_df)] = [location, latitude, longitude, year, outbreak]



    return outbreak_df

# used for checking how long a text would be in chatgpt tokens
def get_tokenized_length(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(text))
    return num_tokens

# returns chunk group of required lengths so as to stay under openai token limit
def build_chunk_group(system_message, text, end_message):
    system_message_length = len(system_message) + len(end_message)
    max_token_length = 4097
    base_multiplier = 4
    safety_multiplier = .8 # used just in case local tokenizer works differently than openai's
    chunk_group = []

    i = 0
    while i < len(text):

        # get initial text length values
        multiplier = base_multiplier
        built_chunk = False
        user_message_length = int(max_token_length * multiplier) - system_message_length
        message = system_message + text[i:i+user_message_length] + end_message
        token_length = get_tokenized_length(message)
        
        # while text is too long for openai model, keep reducing size and try again
        while token_length > int(max_token_length * safety_multiplier):
            multiplier *= .95
            user_message_length = int(max_token_length * multiplier) - system_message_length
            message = system_message + text[i:i+user_message_length] + end_message
            token_length = get_tokenized_length(message)
        
        # add chunk to chunk set, move on
        chunk_group.append(message)
        i += user_message_length

    return chunk_group


#_________________________________________________________________________

# get file name of pdf, create pdf reader instance
pdf_files = glob.glob("*.pdf")
print("Processing all files in this directory. This may take a while!")
for file in pdf_files:
    # if file == 'Navarro et al. 2018 space time SBW.pdf' or file == 'Fraver et al. 2006 time series SBW Maine.pdf' or \
    #     file == "Boulanger et al. 2012 SBW outbreaks 400 yrs.pdf":
    #     continue

    if file != "Blais 1981.pdf":
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
        pdf_text += page.extract_text()

    # set up openai api
    openai_key = "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"
    openai.api_key = openai_key
    model_list = openai.Model.list()
    #print(model_list)
    #input("waiting...")

    # set up dataframe for csv output
    outbreak_df = pd.DataFrame(columns=['area', 'latitude', 'longitude', 'year', 'outbreak'])

    # split data into chunks, set system_message
    system_message_stage_1 = "You are a scientist extracting data from research papers about Spruce Budworm (SBW) infestations and outbreaks. \
                                You are to log every instance in which the text refers to a Spruce Budworm outbreak during any year and region. \
                                You must include specific year and region data. The region must be locatable on a map. General locations like \
                                'study site' or 'tree stand #3' are not relevant. Never include research citations from the text. \
                                It is of the utmost importance that you only output verbatim sentences from the text, \
                                and nothing else."

    system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. \
                                You are to log every instance where the text mentions whether or not an outbreak/infestation \
                                occured during a specific year or range of years and at a specific or general region. \
                                Write every instance in the following format exactly: The location, then the year, whether \
                                there was or was not an outbreak/infestation (always a yes or no), and then a new line. This data must be in csv file \
                                format. Never include the header or any labels. If an outbreak lasts multiple years, write the 'year' feature \
                                as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic \
                                characters. For example, it cannot say 'approximately *year*' or 'unknown'.\
                                It is of the utmost importance that we have as many years \
                                and locations of data as possible. References to other authors and papers are irrelevant. \
                                Only log specific instances of SBW outbreaks."

    end_message = "END\n\n"

    # build prompt chunks (two chunk groups of different slices are built to increase chance that gpt will understand context of text)
    chunk_group = build_chunk_group(system_message_stage_1, pdf_text, end_message)
    chunk_groups = [chunk_group]

    # make api call for each chunk in each chunk_group, print response
    for index, chunk_group in enumerate(chunk_groups):
        j = 0
        while j < len(chunk_group):
            chunk = chunk_group[j]
            user_message = chunk
            temperature = 0
            generated_text = get_chatgpt_response("", user_message, temperature)
            print(f"\nStage 1: {generated_text}")
            generated_text = get_chatgpt_response(system_message_stage_2, generated_text, 0)
            print(f"Stage 2:\n{generated_text}\n\n")
            outbreak_df = parse_response(generated_text, outbreak_df)
            j += 1

    # if there was data to be found, write a new csv file
    if not outbreak_df.empty:
        file_name_no_extension = os.path.splitext(file)[0]
        csv_file_name = 'outbreak_data_' + file_name_no_extension + '.csv'
        outbreak_df.to_csv(csv_file_name, index=False)