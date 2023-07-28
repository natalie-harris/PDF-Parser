import tiktoken

def get_tokenized_length(text, model, examples=[]):

    for example in examples:
        text += example["content"]

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

stage_2_few_shot_examples = [
                    {"role": "user", "content": "Specific SBW Outbreaks:\n\n1. Outbreak: O1 (1905-1930)\n-Region: Eastern Quebec, east of the St. Lawrence River\n\n2. Outbreak: O2 (1935-1965)\n  -Region: Northern reach of southern Quebec\n\n3. Outbreak: O3 (1968-1988)\n-Region: Southwestern Quebec, along a southwest-northeast transect across central southern Quebec"},
                    {"role": "assistant", "content": 'Stage 2:\n"Eastern Quebec, east of the St. Lawrence River", "1905-1930", "Yes"\n"Northern reach of southern Quebec", "1935-1965", "Yes"\n"Southwestern Quebec, along a southwest-northeast transect across central southern Quebec", "1968-1988", "Yes"'},
                    {"role": "user", "content": "Stage 1:\n-Outbreak: 1976-1991\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1946-1959\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1915-1929\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1872-1903\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1807-1817\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1754-1765\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1706-1717\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1664-1670\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1630-1638\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1647-1661 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1606-1619 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec\n-Outbreak: 1564-1578 (uncertain)\n-Region: Northern limit of the temperate forest in southern Quebec"},
                    {"role": "assistant", "content": '"Northern limit of the temperate forest in southern Quebec", "1976-1991", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1946-1959", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1915-1929", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1872-1903", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1807-1817", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1754-1765", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1706-1717", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1664-1670", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1630-1638", "Yes"\n"Northern limit of the temperate forest in southern Quebec", "1647-1661", "Uncertain"\n"Northern limit of the temperate forest in southern Quebec", "1606-1619", "Uncertain"\n"Northern limit of the temperate forest in southern Quebec", "1564-1578", "Uncertain"'},
                    {"role": "user", "content": "Specific SBW outbreaks in specific years and locations mentioned in the text are as follows:\n\n1. Outbreak in 1989: Two epicenters were identified during this outbreak.\n2. Outbreak in 1991: Four epicenters were identified during this outbreak.\n3. Outbreak in 1996: One epicenter was identified during this outbreak.\n4. Outbreak in 2006: Four epicenters were identified during this outbreak.\n5. Outbreak in 2011: One epicenter was identified during this outbreak.\n\nPlease note that the specific locations of these epicenters were not provided in the text."},
                    {"role": "assistant", "content": "No specific instances of SBW outbreaks with geographic locations and years were mentioned in the text."}
                ]

print(get_tokenized_length('', 'gpt-3.5-turbo', stage_2_few_shot_examples))

