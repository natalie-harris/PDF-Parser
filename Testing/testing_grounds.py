"""
Used for figuring out how to get chatgpt to ignore bad data (aka data that does not include specific location/year information)

1. Import libraries
2. Setup function defs
3. Setup input text
4. Setup prompts
5. Test the same input on multiple different prompts
"""

# Import libraries
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

def get_tokenized_length(text, model):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# helper function to get chatgpt output from set of inputs
# note on temperature from OpenAI: "OpenAI models are non-deterministic, meaning that identical inputs can yield different outputs. Setting temperature to 0 will make the outputs mostly deterministic, but a small amount of variability may remain."
# see https://community.openai.com/t/the-system-role-how-it-influences-the-chat-behavior/87353 for suggested ideal system message placement
def get_chatgpt_response(system_message, user_message, temp, use_gpt4=False):
    gpt_model = ''
    system_message = str(system_message)
    user_message = str(user_message)
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

#________________________________________________________________________________________________________________________________________________________________

# set up openai api
openai_key = "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"
openai.api_key = openai_key
model_list = openai.Model.list()

# list of lists of inputs
# each sublist is [0]: system message, [1]: body of text, [2]: end message
system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbrea/infestation occured during a specific year or range of years and at a specific geographic location.\n\nPresent your findings in the following consistent format: '\"Geographic location\"', '\"Year or Year range\"', '\"Outbreak presence (Yes/No/Uncertain)\"'.\n\nFor each instance, output should be a new line in this format, with no headers or labels included.\n\nThe geographic location must be identifiable on a map and can be a city, county, specific lake, etc. Do not include nonspecific or nonidentifiable locations like 'study site'.\n\nIf an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'.\n\nIf the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'.\n\nIt is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks.\n",
end_message = '  END\n\n'
stage_2_inputs = [
    [   # gpt4
        system_message_stage_2,
        "1. Outbreak in southern Quebec, Canada (1976-1991)\n2. Outbreak in southern Quebec, Canada (1946-1959)\n3. Outbreak in southern Quebec, Canada (1915-1929)\n4. Outbreak in southern Quebec, Canada (1872-1903)\n5. Outbreak in southern Quebec, Canada (1807-1817)\n6. Outbreak in southern Quebec, Canada (1754-1765)\n7. Outbreak in southern Quebec, Canada (1706-1717)\n8. Outbreak in southern Quebec, Canada (1664-1670)\n9. Outbreak in southern Quebec, Canada (1630-1638)\n10. Uncertain outbreak in southern Quebec, Canada (1647-1661)\n11. Uncertain outbreak in southern Quebec, Canada (1606-1619)\n12. Uncertain outbreak in southern Quebec, Canada (1564-1578)\n1) Outbreak between 1976 and 1986 in the A03, A06, and A07 old-growth stands.\n2) Outbreak between 1943 and 1959 in the A03 old-growth stand.\n3) Outbreak between 1915-1929 (O3) and 1872-1903 (O4) in the study area.\n4) Outbreaks between 1807-1817 (O5), 1754-1765 (O6), and 1706-1717 (O7) in the study area.\n5) Outbreaks during the 17th century in the study area.\n6) Outbreaks between 1564 and 1590 in the Bmont and ESP chronologies.\n7) Outbreaks during the 1569-1577 and 1584-1586 intervals in the study area.\n8) Outbreaks between 1600-2000 in southern Quebec.\n9) Outbreaks between 1706-1915 in southern Quebec.\n10) Outbreaks prior to 1706 in southern Quebec.\n11) Outbreaks during the 20th century in old-growth stands located in the southernmost part of the study area.\n12) Outbreaks prior to O7 were only reconstructed from sites located along the St. Lawrence River.",
        end_message
    ],
    [   # gpt4
        system_message_stage_2,
        "-1986-2014, Province of Québec, Canada\n-1986-2014, 625 000 km2 territory located in the province of Québec, Canada\n-1986-2014, 1 degree × 1 degree grid covering the entire study area in Québec, Canada\n-1991-2011, Six separate epicenters in Québec, Canada, each separated by a distance of >50 km.\n1. Outbreak in Québec, Canada -1986-2014\n2. Outbreak in six epicenters in Québec, Canada -1989 (two epicenters), 1991 (four epicenters), 1996 (one epicenter), 2006 (four epicenters), and 2011 (one epicenter)",
        end_message
    ],
    [   # gpt4
        system_message_stage_2,
        "1. 1937-1955: Severe infestations of the spruce budworm in the Lac Seul and Lake Nipigon regions of northwestern Ontario.\n2. 1939: First severe defoliation within the epicenters of the infestations probably occurred in the Lac Seul region.\n3. 1937 or 1938: First severe defoliation within the epicenters of the infestations probably occurred in the Lake Nipigon region.\n4. 1862-1872: Evidence from suppression patterns in the annual rings of old-age white spruce indicates that a severe infestation was active in the Lac Seul area.\n5. 1800-1810: Evidence indicates that the entire Lake Nipigon basin was affected.\n6. 1880-1890: A more restricted area southwest of Lake Nipigon was affected.\n7. 1925 and 1926: An infestation in the Lac des Mille Lacs area (about 60 miles west of Port Arthur).\n8. 1943: Two well-developed infestations of the spruce budworm in Sioux Lookout Region and Lake Nipigon-Geraldton Region.\n9. 1945: A notable extension of the spruce budworm outbreak in western Ontario was discovered independently by foresters of the Ontario-Minnesota Pulp and Paper Co., and by forest insect rangers, in late July and early August.\n10. 1946-1955: The Lac Seul infestation expanded rapidly after its discovery in 1945.\n11. 1944-1955: The Lake Nipigon Infestation expanded steadily from its discovery in 1943 until 1948.\n12. 1948: New areas of heavy defoliation were discovered to the south of the main area on Sibley and Black Bay peninsulas and between Gurney and Pays Plat on the north shore of Lake Superior.\n13. 1949: The Lake Nipigon infestation, which has been exceedingly active during the past six or seven years, has shown a marked decline in practically all areas.\n1. 1943 -Sioux Lookout-Hudson infestation area, Northwestern Ontario\n2. 1945 -Lac Seul infestation area, Northwestern Ontario\n3. 1940 or 1938 -Chief Bay area, Lake Nipigon, Northwestern Ontario\n4. 1940 or 1938 -Gull Bay area, Lake Nipigon, Northwestern Ontario\n5. 1939 -Italian Bay of Lost Lake, Northwestern Ontario\n6. 1942 -Gull Bay-Chief Bay epicentre, Lake Nipigon, Northwestern Ontario\n7. 1943 -Lake Nipigon infestation area, Northwestern Ontario\n8. 1948 -Sibley and Black Bay peninsulas, Northwestern Ontario\n9. 1948 -Gurney-Pays Plat area, Northwestern Ontario\n10. 1948 -Lake-of-the-Woods infestation area, Northwestern Ontario\n11. 1952 -Sturge-Disraeli-Little Sturge Lakes area, Northwestern Ontario\n12. 1953 -East of Gurney-Pays Plat area, Northwestern Ontario\n13. 1955 -South of Black Sturgeon Lake in the Port Arthur District, Northwestern Ontario\n14. 1955 -West of Leckie Lake on the southwestern border of the area, Northwestern Ontario\n15. 1955 -Sibley and Black Bay peninsulas, Northwestern Ontario\n16. 1955 -Northern shore of Nipigon Bay east to Rossport and North to Greenhedge Lake in the Geraldton District, Northwestern Ontario\n17. 1955 -North of Greenhedge Lake, Northwestern Ontario\n18. 1955 -Bell Township on the eastern border of the District, Northwestern Ontario\n19. 1955 -Lake Nipigon basin, Northwestern Ontario\n20. 1955 -Lac Seul and Lake Nipigon infestations area, Northwestern Ontario\n1. Lake Nipigon and Sioux Lookout-Hudson infestation, 1943\n2. Sibley-Black Bay area, Pie Island, 1946\n3. Sibley Peninsula, 1947\n4. Central portion of the Sibley Peninsula and throughout all of Black Bay Peninsula, 1958\n5. Gurney-Pays Plat area, 1947-1948\n6. Lake-of-the-Woods infestation, 1947-1948\n7. Main Lake Nipigon infestation, 1949-1955\n8. Lac Seul infestation, 1952\n9. Sioux Lookout area, 1946-1951\n10. Little Sturge Lake area, 1945\n11. Eaglehead Lake area, 1946\n12. Circle Lake area, 1945 (no post-spray checks were made)\n13. Sioux Lookout-Hudson infestation area, 1945 (no post-spray checks were made)",
        end_message
    ],
    [   # gpt-3.5-turbo
        system_message_stage_2,
        "-Spruce budworm infestations in the Lac Seul and Lake Nipigon regions of northwestern Ontario between 1937 and 1955\n-First severe defoliation within the epicenters of the infestations occurred in 1939 in the Lac Seul region and in 1937 or 1938 in the Lake Nipigon region\n-Expansion beyond the epicenters did not occur until the mortality of balsam fir had commenced\n-Increase of the budworm population to the infestation level occurred abruptly\n-Relatively small and scattered representation of the primary hosts gave no protection from severe damage\n-Potential for future losses within the areas affected by 1955 is almost double the amount destroyed\n-Forest cover in the areas described in general terms, with balsam fir and white spruce forming a small part of the volume of the forest cover\n-Estimated that approximately 17 million cords of pulpwood were destroyed due to the infestations\nSpecific SBW outbreaks in specific years and locations mentioned in the text are as follows:\n1. Lake Nipigon and Sioux Lookout-Hudson infestation in 1943: The presence of the budworm in scattered locations such as Hawk Lake near Kenora and Pays Plat on Lake Superior was evident only after intensive searching.\n2. Sibley-Black Bay infestation in 1946: The first indication of defoliation damage in the Sibley-Black Bay area came from Pie Island, immediately southwest of Sibley Peninsula.\n3. Gurney-Pays Plat infestation in 1948: Serious defoliation throughout any appreciable area did not occur until 1948.\n4. Lake-of-the-Woods infestation in 1948: Heavy defoliation was reported over a widespread area in 1948.\n5. Sibley-Black Bay infestation from 1948 onwards: The infestation increased in both area and intensity from 1948 onwards.\n6. Lake Nipigon infestation decline in 1949: The main Lake Nipigon infestation declined as a unit, even in areas where defoliation had occurred for only one to three years.\n7. Main Lac Seul infestation from 1952 onwards: The infestation developed as numerous patches that later coalesced, spreading outwards year after year.\nPlease note that the specific locations mentioned in the text, such as Lake Nipigon, Sioux Lookout-Hudson, Sibley-Black Bay, Gurney-Pays Plat, and Lake-of-the-Woods, can be located on a map.",
        end_message
    ]
]

models = ['gpt-3.5-turbo', 'gpt-4']
use_gpt4 = False

def main():

    for inputs in stage_2_inputs:
        system_message = inputs[0]
        user_message = inputs[1]
        end_message = inputs[2]
        temperature = 0

        print('\n\n' + user_message + '\n')

        for model in models:
            if model == 'gpt-4':
                use_gpt4 = True
            else:
                use_gpt4 = False
                
            print(f"\n\nModel: {model}\n")

            generated_text = get_chatgpt_response(system_message, user_message + end_message, temperature, use_gpt4)
            print(generated_text + '\n\n')


if __name__ == "__main__":
    main()