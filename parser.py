"""
Natalie Harris, NIMBioS
5/23/23

This program extracts text from downloaded pdf, feeds it and a system_message to openai api, and retrieves information parsed by chatgpt

"""

import os
import time
import socket
import PyPDF2
import openai
import tiktoken
import glob
from geopy.geocoders import Nominatim
from geopy.adapters import AdapterHTTPError
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

        except openai.error.ServiceUnavailableError as err:
            num_seconds = 3
            print(f"Server overloaded. Waiting {num_seconds} and retrying request.")

def get_latitude_longitude(town, state, country):
    geolocater = Nominatim(user_agent = "research_paper_parser")
    has_coords = False
    while not has_coords:
        try:
            time.sleep(1)
            location = geolocater.geocode(query=f"{town}, {state}, {country}")
            has_coords = True
        except (AdapterHTTPError, socket.timeout) as err:
            print("There was an HTTP error getting coordinates. Retrying...")
        
    
    if location is None:
        return None, None # must handle when location is not found

    latitude = location.latitude
    longitude = location.longitude

    return latitude, longitude

def location_to_coordinates(location, system_message):
    user_message = location 
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
    split_line = original_line.split(',')
    location = split_line[0].strip()
    years = split_line[1].strip()
    outbreak = split_line[2].strip()
    source = split_line[3] .strip()
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
        strings = [location, str(i), outbreak, source]
        new_line = ", ".join(strings)
        new_list.append(new_line)

    return new_list


# parse gpt output to convert location to lat/long and reduce output clutter
def parse_response(response, outbreak_df, system_message_stage_3):
    split_response = response.splitlines()
    new_split_response = [] # used for adding additional lines when chatgpt gives range of years

    # cache previously found location coordinates to reduce chatgpt use
    cached_location_coords = {}

    # for each line in chatgpt's response
    for line in split_response:

        # make sure line is in the correct format, otherwise move to next line
        line = make_csv_format(line)
        print(f"Slightly more formatted: {line}")
        split_line = line.split(',')

        if len(split_line) != 4:
            continue

        location = split_line[0].lower().strip()
        year = split_line[1].lower().strip()
        outbreak = split_line[2].lower().strip()
        source = split_line[3].lower().strip()


        if outbreak != 'yes' and outbreak != 'no' and outbreak != 'uncertain':
            continue
        if any(char.isalpha() for char in year):
            continue
        if len(year) != 4 and len(year) != 9:
            continue
        if len(location) <= 3:
            continue
        if len(source) <= 3:
            continue

        # if data given as range of years, add every year to new list
        print(year)
        print(len(year))
        if len(year) == 9 and year[4] == '-':
            every_year = list_each_year(line)
            print(every_year[0])
            if len(every_year) > 1:
                print(every_year)
                for single_year in every_year:
                    new_split_response.append(single_year)
        else:
            new_split_response.append(line)

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
                return outbreak_df
            cached_location_coords[location] = (latitude, longitude)
        
        # add latitude and longitude to dataframe
        split_line.append(latitude)
        split_line.append(longitude)
        print(f"{location}, {latitude}, {longitude}, {year}, {outbreak}, {source}")
        outbreak_df.loc[len(outbreak_df)] = [location, latitude, longitude, year, outbreak, source]



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

file_name = "Testing/testing_data/test3"

# set system_messages for each stage
system_message_stage_0 = "You are a list-maker making a comma-separated list of sources for research papers about spruce budworms. \
                            You are given an excerpt from the text and must determine where the data is coming from. Your possible list items are: \
                            Dendrochronological samples from tree cores, Dendrochronological samples from historical buildings, \
                            Pheromone traps, Aerial defoliation survey, Survey from insect laboratory, or Personal Communication with \
                            the Department of Lands and Forest representative. If the paper uses multiple sources, list each one separately, \
                            using commas as delimiters. If no information about the \
                            methods of data collection are given, simple output 'Unknown'. It is of the utmost importance that your output \
                            is a comma-separated list. Do not write headers or any additional information. Preface \
                            the information with 'Data collection method: '."

system_message_stage_1 = "You are a scientist extracting data from research papers about Spruce Budworm (SBW) infestations \
                            and outbreaks. You are to log every instance in which the text refers to a Spruce Budworm outbreak \
                            during any years and region. You must include the range of years, the specific region data, and the origin of \
                            the data for the outbreak came from. Valid origins are: Dendrochronological samples from tree cores, \
                            Dendrochronological samples from historical buildings, Pheromone traps, Aerial defoliation survey, \
                            Survey from insect laboratory, or Personal Communication with the Department of Lands and Forest \
                            representative. The \
                            region must be locatable on a map. Be as specific as possible. General locations like 'study site' \
                            or 'tree stand #3' are not relevant. Include outbreaks whose existence is uncertain. Never include \
                            research citations from the text. It is of the utmost importance that you only output verbatim \
                            sentences from the text, and nothing else."

system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. \
                            You are to log every instance where the text mentions whether or not an outbreak/infestation \
                            occured during a specific year or range of years and at a specific geographic location. Write every \
                            instance in the following format exactly: The geographic location, then the year, whether there was or was \
                            not an outbreak/infestation (always a yes or no), the origin of the data, and then \
                            a new line. This data must be in csv \
                            file format. Never include the header or any labels. Valid origins for the data are: Dendrochronological \
                            samples from tree cores, \
                            Dendrochronological samples from historical buildings, Pheromone traps, Aerial defoliation survey, \
                            Survey from insect laboratory, or Personal Communication with the Department of Lands and Forest \
                            representative. The geographic location must be something like a city, \
                            a county, a specific lake, or anything that is locatable on a map. If an outbreak lasts multiple years, \
                            write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The \
                            year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' \
                            or 'unknown'. It is of the utmost importance that we have as many years and locations of data as \
                            possible. References to other authors and papers are irrelevant. Only log specific instances of \
                            SBW outbreaks. If the authors are uncertain of an outbreak's existence, the 'outbreak' column for \
                            that outbreak should be 'uncertain'"

system_message_stage_3 = "You are a computer made to give scientists town names within an area. You will be given a location \
                            in North America. Your task is to give a town that belongs at that location to be used as a \
                            locality string for GEOLocate software. If the area is very remote, give the nearest town. Put \
                            it in csv format as the following: \"city, state, country\". It is of the utmost importance that \
                            you print only the one piece of data, and absolutely nothing else. You must output a city name, \
                            even if the given area is very large or very remote."

end_message = "END\n\n"

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

# for concatenating dataframes
data_list = []

# get folder path and file name of pdf, create pdf reader instance
pdf_files = glob.glob("papers/*.pdf")
print("Processing all files in this directory. This may take a while!")
for file in pdf_files:

    # if file != 'papers/Boulanger et al. 2012 SBW outbreaks 400 yrs.pdf':
    #     continue

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

    # test source determination
    # build chunks
    source_chunk_group = build_chunk_group(system_message_stage_0, pdf_text, end_message)
    source_chunk_groups = [source_chunk_group]
    source_prefix = "data collection method: "

    # iterate through each chunk until source is found
    source = 'unknown'
    for index, chunk_group in enumerate(source_chunk_groups):
        j = 0
        while j < len(chunk_group) and source == 'unknown':
            chunk = chunk_group[j]
            user_message = chunk
            temperature = 0
            generated_text = get_chatgpt_response("", user_message, temperature).lower()
            if generated_text.startswith(source_prefix):
                generated_text = generated_text[len(source_prefix):]
            if generated_text.endswith('.'):
                generated_text = generated_text[0:len(generated_text) - 1]
            if not generated_text.startswith('unknown'):
                print(generated_text)
                source = generated_text
            j += 1

    valid_source_outputs = []
    source = source.split(',')
    for i in range(len(source)):
        source[i] = source[i].strip().lower()
        if source[i] in valid_sources:
            valid_source_outputs.append(source[i])
    
    print(valid_source_outputs)

    # set up dataframe for csv output
    outbreak_df = pd.DataFrame(columns=['area', 'Latitude', 'Longitude', 'Year', 'Outbreak', 'Source'])

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
            outbreak_df = parse_response(generated_text, outbreak_df, system_message_stage_3)
            j += 1

    

    # if there was data to be found, add it to dataframe list
    if not outbreak_df.empty:
        outbreak_df['File Name'] = os.path.basename(file)
        outbreak_df['Study'] = outbreak_df['File Name'].map(study_indices)

        # Append the dataframe to the list
        data_list.append(outbreak_df)

        # Create individual csv file for this study
        # file_name_no_extension = os.path.splitext(file)[0]
        # csv_file_name = 'outbreak_data_' + file_name_no_extension + '.csv'
        # excel_file_name = 'outbreak_data_' + file_name_no_extension + '.xlsx'
        # outbreak_df.to_csv(csv_file_name, index=False)
        # outbreak_df.to_excel(excel_file_name, index=False)

# concatenate all dataframes
final_data = []
for df in data_list:

    data = df
    filename = data['File Name'].iloc[0]
    study = data['Study'].iloc[0]
    list_data_filled = []
    data = data.sort_values(['area', 'Year'])
    
    for area in data['area'].unique():
        area_data = data[data['area'] == area].copy()
        
        # Convert 'Year' column to int
        area_data['Year'] = area_data['Year'].astype(int)
        
        min_year = int(area_data['Year'].min())
        max_year = int(area_data['Year'].max())
        latitude = area_data['Latitude'].iloc[0]
        longitude = area_data['Longitude'].iloc[0]
        
        all_years = pd.DataFrame({'Year': range(min_year - 1, max_year + 2)})
        all_years['area'] = area
        all_years['Latitude'] = latitude
        all_years['Longitude'] = longitude
        
        # Convert 'area', 'Latitude', and 'Longitude' in both DataFrames to the same data type if needed
        # e.g., all_years['area'] = all_years['area'].astype(str)
        #       area_data['area'] = area_data['area'].astype(str)
        
        merged_data = pd.merge(all_years, area_data, how='left', on=['Year', 'area', 'Latitude', 'Longitude'])
        merged_data['Outbreak'].fillna('no', inplace=True)
        merged_data['Study'] = study
        merged_data['File Name'] = filename
        list_data_filled.append(merged_data)

    data = pd.concat(list_data_filled, ignore_index=True)

    data['Outbreak'] = data['Outbreak'].map(outbreak_occurence_values)
    final_data.append(data)

if len(final_data) > 0:
    all_data = pd.concat(final_data, ignore_index=True)
    all_data.to_csv(file_name + '.csv', index=False)
    all_data.to_excel(file_name + '.xlsx', index=False)
