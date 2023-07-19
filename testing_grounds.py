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

# list of lists of inputs
# each sublist is [0]: system message, [1]: body of text, [2]: end message
system_message_stage_2 = "You are a computer analyzing a text for scientists on spruce budworm (SBW) outbreaks/infestations. You are to log every instance where the text mentions whether or not an outbrea/infestation occured during a specific year or range of years and at a specific geographic location.\n\nPresent your findings in the following consistent format: '\"Geographic location\"', '\"Year or Year range\"', '\"Outbreak presence (Yes/No/Uncertain)\"'.\n\nFor each instance, output should be a new line in this format, with no headers or labels included.\n\nThe geographic location must be identifiable on a map and can be a city, county, specific lake, etc. Do not include nonspecific or nonidentifiable locations like 'study site'.\n\nIf an outbreak lasts multiple years, write the 'year' feature as 'first_year-last_year'. There MUST be a dash in between the two years. The year section must have no alphabetic characters. For example, it cannot say 'approximately *year*' or 'unknown'.\n\nIf the authors are uncertain of an outbreak's existence, the 'outbreak' column for that outbreak should be 'uncertain'.\n\nIt is of the utmost importance that we have as many years and locations of data as possible. References to other authors and papers are irrelevant. Only log specific instances of SBW outbreaks.\n",
end_message = '  END\n\n'
stage_2_inputs = [
    [   
        system_message_stage_2,
        "1. Outbreak in southern Quebec, Canada (1976-1991)\n2. Outbreak in southern Quebec, Canada (1946-1959)\n3. Outbreak in southern Quebec, Canada (1915-1929)\n4. Outbreak in southern Quebec, Canada (1872-1903)\n5. Outbreak in southern Quebec, Canada (1807-1817)\n6. Outbreak in southern Quebec, Canada (1754-1765)\n7. Outbreak in southern Quebec, Canada (1706-1717)\n8. Outbreak in southern Quebec, Canada (1664-1670)\n9. Outbreak in southern Quebec, Canada (1630-1638)\n10. Uncertain outbreak in southern Quebec, Canada (1647-1661)\n11. Uncertain outbreak in southern Quebec, Canada (1606-1619)\n12. Uncertain outbreak in southern Quebec, Canada (1564-1578)\n1) Outbreak between 1976 and 1986 in the A03, A06, and A07 old-growth stands.\n2) Outbreak between 1943 and 1959 in the A03 old-growth stand.\n3) Outbreak between 1915-1929 (O3) and 1872-1903 (O4) in the study area.\n4) Outbreaks between 1807-1817 (O5), 1754-1765 (O6), and 1706-1717 (O7) in the study area.\n5) Outbreaks during the 17th century in the study area.\n6) Outbreaks between 1564 and 1590 in the Bmont and ESP chronologies.\n7) Outbreaks during the 1569-1577 and 1584-1586 intervals in the study area.\n8) Outbreaks between 1600-2000 in southern Quebec.\n9) Outbreaks between 1706-1915 in southern Quebec.\n10) Outbreaks prior to 1706 in southern Quebec.\n11) Outbreaks during the 20th century in old-growth stands located in the southernmost part of the study area.\n12) Outbreaks prior to O7 were only reconstructed from sites located along the St. Lawrence River.",
        end_message
    ]
]

def main():

    print("main")

if __name__ == "__main__":
    main()