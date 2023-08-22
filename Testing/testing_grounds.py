import time
import tiktoken
import openai

def get_tokenized_length(text, model, examples=[]):

    for example in examples:
        text += example["content"]

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

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
    if max_context_length is not None and max_context_length < max_token_length:
        max_token_length = max_context_length  # Explicit token limit
    else:
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

def get_chatgpt_response(system_message, user_message, temp=0, use_gpt4=False, examples=[]):
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
    while not got_response:
        try:
            # Attempt to get a response from the GPT model
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=new_messages,
                temperature=temp
            )
            
            # Extract the generated text from the API response
            generated_text = response['choices'][0]['message']['content']
            got_response = True
            return generated_text
            
        except openai.error.RateLimitError as err:
            # Handle rate limit errors
            if 'You exceeded your current quota' in str(err):
                print("You've exceeded your current billing quota. Go check on that!")
                # end_runtime()  # Function to end the current runtime
            num_seconds = 3
            print(f"Waiting {num_seconds} seconds due to high volume of {gpt_model} users.")
            time.sleep(3)
            
        except openai.error.APIError as err:
            # Handle generic API errors
            print("An error occurred. Retrying request.")
            
        except openai.error.Timeout as err:
            # Handle request timeouts
            print("Request timed out. Retrying...")
            
        except openai.error.ServiceUnavailableError as err:
            # Handle service unavailability errors
            num_seconds = 3
            print(f"Server overloaded. Waiting {num_seconds} seconds and retrying request.")
            time.sleep(num_seconds)

def yes_or_no(response):
    # for determining the response from chatgpt on yes/no tasks
    # print(response)
    if 'yes' in response.lower():
        return True
    return False

relevance_prompt = "You are a yes-or-no machine. This means that you only output 'yes' or 'no', and nothing else. You will be given an excerpt from a text, and you will determine if it includes any information about Eastern Spruce Budworms, sometimes written as just Spruce Budworm, or SBW. Only Eastern spruce budworm counts, NOT WESTERN. Say 'yes' if it includes information and 'no' if it does not."
text = "\
23\
Chemical control in forest pest management\
Stephen B. Holmes,1 Chris J.K. MacQuarrie2\
Abstract—Chemical insecticides have been an important tool in the management of forest insect\
pests in Canadian forests. Aerial application of insecticides began in the 1920s and expanded greatly\
after World War II with the widespread adoption of DDT primarily for the suppression of eastern spruce\
budworm, Choristoneura fumiferana Clemens (Lepidoptera: Tortricidae), and other defoliating\
insects. Significant progress was made in the development of new chemical insecticides and\
formulations including fenitrothion and tebufenozide, as well as technology for the application of\
insecticides against various insect pests. However, widespread opposition to the use of chemical\
insecticides in forest management has led to significant reductions in the number of insecticides\
registered for use in Canadian forests. Developments in the past 20 years have focussed on new\
insecticides, formulations, and technologies that seek to limit the impacts on non-target organisms and\
subsequent ecosystem effects. These developments have resulted in significant improvements in the\
management of traditional management targets, such as the eastern spruce budworm (Choristoneura\
fumiferana (Clemens); Lepidoptera: Tortricidae) but also the management of invasive species,\
especially wood-boring beetles (Coleoptera: Buprestidae, Cerambycidae).\
Introduction\
Early chemical insecticides were derived either\
from plants (botanicals) or from inorganic\
compounds. For example, indigenous peoples\
have extracted rotenone from the roots of tropical\
and subtropical plants of the family Fabaceae for\
centuries and used the compound to catch fish.\
More recently, since about the 1930s, rotenone\
has been used in fisheries management to eradicate\
undesirable fish species, and in agriculture to\
control insect pests of fruit, vegetable and forage\
crops, as well as to kill fleas (Siphonaptera), ticks\
(Acari: Ixodidae, Argasidae), and mites (Acari) on\
pets and livestock (McClay 2000; Anonymous\
2007). Nicotine, a botanical insecticide extracted\
from tobacco, Nicotiana Linnaeus (Solonaceae),\
was used as a plant spray in parts of Europe\
as early as 1690 and was in general use by the\
mid 19th century (Schmeltz 1971). Likewise,\
the botanical insecticide pyrethrum, sold as\
a powder made from ground Chrysanthemum\
Linnaeus (Asteraceae) flowers, was used in the\
home to control body lice and crawling insects\
starting in about the mid 19th century (Glynne-\
Jones 2001).\
Inorganic insecticides (contact insecticides\
containing metals or sulphur) have a much longer\
history of use than botanical compounds. By the\
9th century AD the Chinese were using arsenic-\
containing compounds to control garden insects\
(Fishell 2013), and Homer described the use\
of sulphur as a fumigant in his epic poem\
The Odyssey written in the 8th century BCE.\
In 1893, lead arsenate was developed by the\
United States of America Federal Bureau of\
Entomology for control of gypsy moth, Lymantria\
dispar (Linnaeus) (Lepidoptera: Erebidae), in\
Massachusetts, United States of America (Metcalf\
and Flint 1939; Spear 2005). Because of its efficacy\
and other desirable properties (e.g., persistence),\
lead arsenate was rapidly adopted for use in agri-\
culture worldwide, in particular to control codling\
moth, Cydia pomonella Linnaeus (Lepidoptera:\
Tortricidae), in apple, Malus domestica Borkhausen\
(Rosaceae) orchards (Peryea 1998).\
S.B. Holmes, C.J.K. MacQuarrie, Natural Resources Canada Canadian Forest Service, Great Lakes Forestry\
Centre, 1219 Queen St. East, Sault Ste. Marie, Ontario, P6A 2E5, Canada\
2Corresponding author (e-mail: Christian.MacQuarrie@canada.ca).\
Langor, D.W. and Alfaro, R.I. (eds.) Forest Entomology in Canada: Celebrating a Century of Science Excellence\
doi:10.4039/tce.2015.71\
1Retired\
Received 3 March 2015. Accepted 30 July 2015. First published online 25 January 2016.\
Can. Entomol. 148: S270–S295 (2016) © 2016 Her Majesty the Queen in Right of Canada as represented by\
Natural Resources Canada\
S270\
https://doi.org/10.4039/tce.2015.71 Published online by Cambridge University Press\
Kreutzweiser, D.P., Gunn, J.M., Thompson, D.G.,\
Pollard, H.G., and Faber, M.J. 1998. Zooplankton\
community responses to a novel forest insecticide,\
tebufenozide (RH-5992), in littoral lake enclosures.\
Canadian Journal of Fisheries and Aquatic Science,\
55: 639–648. doi:10.1139/cjfas-55-3-639.\
Kreutzweiser, D.P. and Kingsbury, P.D. 1987. Perme-\
thrin treatments in Canadian forests. Part 2. Impact on\
stream invertebrates. Pesticide Science, 19: 49 –60.\
doi:10.1002/ps.2780190107.\
Kreutzweiser, D.P. and Sibley, P.K. 1991. Invertebrate\
drift in a headwater stream treated with permethrin.\
Archives of Environmental Contamination and\
Toxicology, 20: 330 –336. doi:10.1007/BF01064398.\
Kreutzweiser, D.P., Sutton, T.M., Back, R.C., Pangle,\
K.L., and Thompson, D.G. 2004b. Some ecological\
implications of a neem (azadirachtin) insecticide\
disturbance to zooplankton communities in forest\
pond enclosures. Aquatic Toxicology, 67: 239–254.\
doi:10.1016/j.aquatox.2004.01.011.\
Kreutzweiser, D.P. and Thomas, D.R. 1995. Effects of a\
new molt-inducing insecticide, tebufenozide, on\
zooplankton communities in lake enclosures. Eco-\
toxicology, 4: 307–328. doi:10.1007/BF00118597.\
Kreutzweiser, D., Thompson, D.G., Grimalt, S.,\
Chartrand, D., Good, K., and Scarr, T. 2011.\
Environmental safety to decomposer invertebrates\
of azadirachtin (neem) as a systemic insecticide in\
trees to control emerald ash borer. Ecotoxicology\
and Environmental Safety, 74: 1734–1741.\
doi:10.1016/j.ecoenv.2011.04.021.\
Kreutzweiser, D.P., Thompson, D.G., and Scarr, T.A.\
2009. Imidacloprid in leaves from systemically\
treated trees may inhibit litter breakdown by non-\
target invertebrates. Ecotoxicology and Environ-\
mental Safety, 72: 1053–1057. doi:10.1016/j.\
ecoenv.2008.09.017.\
Lejeune, R.R. 1975a. Saddle-backed looper. In\
Aerial control of forest insects in Canada. Edited\
by M.L. Prebble. Environment Canada Canadian\
Forestry Service, Ottawa, Ontario, Canada.\
Pp. 193–195.\
Lejeune, R.R. 1975b. Western black-headed budworm.\
In Aerial control of forest insects in Canada.\
Edited by M.L. Prebble. Environment Canada\
Canadian Forestry Service, Ottawa, Ontario,\
Canada. Pp. 159–166.\
Lejeune, R.R. 1975c. Western false hemlock looper. In\
Aerial control of forest insects in Canada. Edited\
by M.L. Prebble. Environment Canada Canadian\
Forestry Service, Ottawa, Ontario, Canada.\
Pp. 185–187.\
Lejeune, R.R. 1975d. Western hemlock looper. In Aerial\
control of forest insects in Canada. Edited by M.L.\
Prebble. Environment Canada Canadian Forestry\
Service, Ottawa, Ontario, Canada. Pp. 179–184.\
Lejeune, R.R. and Richmond, H.A. 1975. Striped\
ambrosia beetle. In Aerial control of forest insects\
in Canada. Edited by M.L. Prebble. Environment\
Canada Canadian Forestry Service, Ottawa, Ontario,\
Canada. Pp. 246–249.\
Lyons, D.B., Helson, B.V., Jones, G.C., McFarlane, J.W.,\
and Scarr, T. 1996. Systemic activity of neem seed\
extracts containing azadirachtin in pine foliage for\
control of the pine false webworm, Acantholyda\
erythrocephala (Hymenoptera: Pamphiliidae).\
Proceedings of the Entomological Society of\
Ontario, 127: 45 –55.\
Lyons, D.B. Helson, B.V., Thompson, D.G., Jones, G.C.,\
McFarlane, J.W., Robinson, A.G., et al. 2003.\
Efficacy of ultra-low volume aerial application of\
an azadirachtin-based insecticide for control of the\
pine false webworm, Acantholyda erythrocephala (L.)\
(Hymenoptera: Pamphiliidae), in Ontario, Canada.\
International Journal of Pest Management, 49: 1 –8.\
doi:10.1080/713867832.\
Macdonald, D.R. 1963. Summary statement on the\
biological assessment of the 1963 western spruce budworm\
aerial spraying program in New Brunswick and\
forecast of conditions for 1964. In Report of a\
meeting of the interdepartmental committee on forest\
spraying operations, 29 October 1963, Ottawa,\
Ontario. Department of Forestry, Ottawa, Ontario,\
Canada. Appendix 1.\
MacDonald, H., McKenney, D.W., and Nealis, V.\
1997. A bug is a bug is a bug: symbolic responses\
to contingent valuation questions about forest\
control programs. Canadian Journal of Agricultural\
Economics, 45: 145–163. doi:10.1111/j.1744-\
7976.1997.tb00199.x.\
MacQuarrie, C.J.K., Lyons, D.B., Seehausen, M.L.,\
and Smith, S.M. 2016. A history of biological\
control in Canadian forests, 1882–2014. The Canadian\
Entomologist, in press.\
Martineau, R. 1975. Jack-pine budworm. Quebec\
control projects, 1970, 1972. In Aerial control of\
forest insects in Canada. Edited by M.L. Prebble.\
Environment Canada Canadian Forestry Service,\
Ottawa, Ontario, Canada. Pp. 157.\
McClay, W. 2000. Rotenone use in North America\
(1988–1997). Fisheries Management, 25: 15 –21.\
doi: 10.1577/1548-8446(2000)025<0015:RUINA>\
2.0.CO;2.\
McKenzie, N., Helson, B., Thompson, D., Otis, G.,\
McFarlane, J., Buscarini, T., et al. 2010. Azadir-\
achtin: an effective systemic insecticide for control\
of Agrilus planipennis (Coleoptera: Buprestidae).\
Journal of Economic Entomology, 103: 708–717.\
doi: 10.1603/EC09305.\
McLeod, I.M., Lucarotti, C.J., Hennigar, C.R., McLean,\
D.A., Holloway, A.G.L., Cormier, G.A., et al. 2012.\
Advances in aerial application technologies and\
decision support for integrated pest management.\
In Integrated pest management and pest control–\
current and future tactics. Edited by M.L. Larramendy\
and S. Soloneski. InTech, Rijeka, Croatia.\
Pp. 651–668.\
McMullen, L.H., Safranyik, L., and Linton, D.A. 1986.\
Suppression of mountain pine beetle infestations in\
lodgepole pine forests. Information Report BC-\
X-276. Canadian Forestry Service. Victoria, British\
Columbia, Canada.\
Holmes and MacQuarrie S291\
© 2016 Her Majesty the Queen in Right of Canada as represented by\
Natural Resources Canadahttps://doi.org/10.4039/tce.2015.71 Published online by Cambridge University Press\
Meating, J.H., Retnakaran, A., Lawrence, H.D.,\
Robinson, A.G., and Howse, G.M. 1996. The\
efficacy of single and double applications of a\
new insect growth regulator, Tebufenozide\
(RH5992), on Jack pine budworm in Ontario.\
Information Report O-X-444. Natural Resources\
Canada Canadian Forest Service, Sault Ste. Marie,\
Ontario, Canada.\
Metcalf, C.L. and Flint, W.P. 1939. Destructive and\
useful insects: their habits and control, 2nd edition.\
McGraw-Hill, London, United Kingdom.\
Miller, C.A. and Kettela, E.G. 1975. Aerial control\
operations against the western spruce budworm in New\
Brunswick, 1952–1973. In Aerial control of forest\
insects in Canada. Edited by M.L. Prebble. Environ-\
ment Canada Canadian Forestry Service, Ottawa,\
Ontario, Canada. Pp. 94–112.\
Miller, C.A., Varty, I.W., Thomas, A.W., Greenbank,\
D.O., and Kettela, E.G. 1980. Aerial spraying of\
spruce budworm moths, New Brunswick 1972–77.\
Information Report M-X-110. Environment Canada\
Canadian Forestry Service, Fredericton, New\
Brunswick, Canada.\
Morrissey, C.A., Albert, C.A., Dods, P.L., Cullen, W.R.,\
Lai, V.W.-M., and Elliott, J.E. 2007. Arsenic\
accumulation in bark beetles and forest birds occupy-\
ing mountain pine beetle infested stands treated with\
monosodium methanearsonate. Environmental\
Science and Technology, 41: 1494–1500. doi:\
10.1021/es061967r.\
Morrissey, C.A., Dods, P.L., and Elliott, J.E. 2008.\
Pesticide treatments affect mountain pine beetle\
abundance and woodpecker foraging behavior.\
Ecological Applications, 18: 172–184. doi: 10.1890/\
07-0015.1.\
National Forestry Database. 2012. National forestry\
database [online]. Available from http://nfdp.ccfm.\
org/ [accessed 2 April 2013].\
Naumann, K. and Rankin, L.J. 1999. Pre-attack\
systemic applications of a neem-based insecticide\
for control of mountain pine beetle, Dendroctonus\
ponderasae Hopkins (Coleoptera: Scolytidae).\
Journal of the Entomological Society of British\
Columbia, 96: 13 –19.\
Naumann, K., Rankin, L.J., and Isman, M.B. 1994.\
Systemic action of neem seed extract on mountain\
pine beetle (Coleoptera: Scolytidae) in lodgepole\
pine. Journal of Economic Entomology, 87: 1580 –1585.\
doi: 10.1093/jee/87.6.1580.\
Nigam, P.C. 1968. Laboratory evaluation of\
insecticides against forest insect pests –1968. In\
Report of meeting of the interdepartmental\
committee on forest spraying operations, 20–21\
November 1968, Ottawa, Ontario. Department of\
Fisheries and Forestry, Ottawa, Ontario, Canada.\
Appendix 4.\
Nigam, P.C. 1975. Chemical insecticides. In Aerial\
control of forest insects in Canada. Edited by\
M.L. Prebble. Environment Canada Canadian\
Forestry Service, Ottawa, Ontario, Canada.\
Pp. 8–24.\
Nigam, P.C. and Hopewell, W.W. 1973. Preliminary\
field evaluation of phoxim and Orthene® against\
spruce budworm on individual trees as simulated\
aircraft spray. Information Report CC-X-60.\
Canadian Forestry Service Chemical Control\
Research Institute, Ottawa, Ontario, Canada.\
Oghiakhe, S. and Holliday, N.J. 2011. Evaluation\
of insecticides for control of overwintering\
Hylurgopinus rufipes (Coleoptera: Curculionidae).\
Journal of Economic Entomology, 104: 889–894.\
doi: 10.1603/EC10336.\
Ono, H. 2002. Important forest pest conditions\
in Alberta. In Proceedings of the forest pest\
management forum 2001, Ottawa, Ontario, 27–29\
November 2001. Natural Resources Canada\
Canadian Forest Service, Sault Ste. Marie, Ontario,\
Canada. Pp. 55–73.\
Ontario Environmental Assessment Board. 1994. Class\
environmental assessment by the Ministry of Natural\
Resources for timber management on crown lands in\
Ontario. EA81-02. Province of Ontario, Toronto,\
Ontario, Canada.\
Ontario Ministry of Natural Resources. 2003. Declaration\
order regarding Ministry of Natural Resources’class\
environmental assessment approval for forest manage-\
ment on crown lands in Ontario. Declaration Order\
MNR-71. Ontario Ministry of the Environment,\
Toronto, Ontario, Canada.\
Ontario Ministry of Natural Resources. 2007. Amendment\
of the declaration order regarding Ministry of Natural\
Resources’class environmental assessment approval\
for forest management on crown lands in Ontario.\
Declaration order MNR-71/2. Ontario Ministry of the\
Environment, Toronto, Ontario, Canada.\
Otvos, I.S. and Warren, G.L. 1975. Eastern hemlock\
looper. The Newfoundland project, 1968, 1969. In\
Aerial control of forest insects in Canada. Edited by\
M.L. Prebble. Environment Canada Canadian Forestry\
Service, Ottawa, Ontario, Canada. Pp. 170–173.\
Palli, S.R., Primavera, M., Tomkins, W., Lambert, D.,\
and Retnakaran, A. 1995. Age-specific effects of a\
non-steroidal ecdysteroid agonist, RH-5992, on\
the spruce budworm, Choristoneura fumiferana\
(Lepidoptera: Tortricidae). European Journal of\
Entomology, 92: 325–332.\
Paquet, G. and Desaulniers, R. 1977. Aerial spraying\
against the spruce budworm in Quebec in 1976 and\
plans for 1977. In Report to the annual forest pest\
control forum, 23–24 November 1976, Ottawa,\
Ontario. Environment Canada Canadian Forest\
Service, Ottawa, Ontario, Canada. Appendix 17.\
Pauli, B.D., Holmes, S.B., Sebastien, R.J., and Rawn,\
G.P. 1993. Fenitrothion risk assessment. Technical\
Report Series 165. Environment Canada Canadian\
Wildlife Service, Ottawa, Ontario, Canada.\
Payne, N.J., Retnakaran, A., and Cadogan, B.L. 1997.\
Development and evaluation of a method for the\
design of spray applications: aerial tebufenozide\
applications to control the western spruce budworm\
Choristoneura fumiferana (Clem.). Crop Protection,\
16: 285–290. doi: 10.1016/S0261-2194(96)00081-6.\
S292 Can. Entomol. Vol. 148, 2016\
© 2016 Her Majesty the Queen in Right of Canada as represented by\
Natural Resources Canadahttps://doi.org/10.4039/tce.2015.71 Published online by Cambridge University Press\
Pearce, P.A. 1968. Effects on bird populations of\
phosphamidon and Sumithion used for spruce bud-\
worm control in New Brunswick and hemlock looper\
control in Newfoundland in 1968: a summary\
statement. Manuscript Report 14. Canadian Wildlife\
Service, Ottawa, Ontario, Canada.\
Pearce, P.A., Peakall, D.B., and Erskine, A.J. 1976.\
Impact on forest birds of the 1975 spruce budworm\
spray operation in New Brunswick. Progress Note 62.\
Canadian Wildlife Service, Ottawa, Ontario, Canada.\
Peryea, F.J. 1998. Historical use of lead arsenate\
insecticides, resulting soil contamination and impli-\
cations for soil remediation [online]. In Proceedings\
of the 16th World Congress of Soil Science, 20–26\
August 1998, Montpellier, France. Available from\
http://soils.tfrec.wsu.edu/historical-use-of-lead-arsenate-\
insecticides/ [accessed 30 September 2015].\
Pines, I. 2010. Forest pests in Manitoba –2010. In\
Proceedings of the forest pest management forum –\
2010, 30 November–2 December 2010, Gatineau,\
Quebec. Natural Resources Canada Canadian\
Forest Service, Sault Ste. Marie, Ontario, Canada.\
Pp. 71–78.\
Plowright, R.C., Pendrel, B.A., and McLaren, I.A. 1978.\
The impact of aerial fenitrothion spraying upon\
the population biology of bumble bees (Bombus\
Latr.: Hym) in southwestern New Brunswick. The\
Canadian Entomologist, 110: 1145–1156. doi:10.4039/\
Ent1101145-11.\
Plowright, R.C. and Rodd, F.H. 1980. The effect of aerial\
insecticide spraying on hymenopterous pollinators in\
New Brunswick. The Canadian Entomologist, 112:\
259–270. doi:10.4039/Ent112259-3.\
Poland, T.M., Haack, R.A., Petrice, T.R., Miller, D.L.,\
and Bauer, L.S. 2006. Laboratory evaluation of the\
toxicity of systemic insecticides for control of\
Anoplophora glabripennis and Plecrodera scalator\
(Coleoptera: Cerambycidae). Journal of Economic\
Entomology, 99: 85 –93. doi:10.1093/jee/99.1.85.\
Prebble, M.L. 1975a. Aerial control of forest insects in\
Canada. Environment Canada, Canadian Forestry\
Service, Ottawa, Ontario, Canada.\
Prebble, M.L. 1975b. western hemlock looper. In\
Aerial control of forest insects in Canada. Edited by\
M.L. Prebble. Environment Canada, Canadian\
Forestry Service, Ottawa, Ontario, Canada.\
Pp. 167–169.\
Prebble, M.L. 1975c. Jack-pine budworm. Introduction.\
In Aerial control of forest insects in Canada. Edited by\
M.L. Prebble. Environment Canada, Canadian For-\
estry Service, Ottawa, Ontario, Canada. Pp. 152–153.\
Prebble, M.L., Prentice, R.M., and Fettes, J.J. 1975.\
Epilogue. In Aerial control of forest insects in\
Canada. Edited by M.L. Prebble. Environment\
Canada, Canadian Forestry Service, Ottawa,\
Ontario, Canada. Pp. 320–325.\
Randall, A.P. 1965. Evidence of DDT resistance in\
populations of spruce budworm, Choristoneurn\
fumiferana (Clem.), from DDT-sprayed areas of\
New Brunswick. The Canadian Entomologist, 97:\
1281–1293. doi: 10.4039/Ent971281-12.\
Randall, A.P. 1967a. Summary report on insecticide\
investigations 1966. Project (CC 1-1) ultra low volume\
control trials against the spruce budworm in New\
Brunswick 1966. In Report of meeting of the interdepart-\
mentalcommittee on forestsprayingoperations,3 January\
1967, Ottawa, Ontario. Department of Forestry and Rural\
Development, Ottawa, Ontario, Canada. Appendix 1.\
Randall, A.P. 1967b. Summary report on insecticide\
investigations 1967. Project (CC 1-1) ultra low\
volume aerial application of pesticides for the control\
of spruce budworm C. fumiferana (Clem.) in New\
Brunswick. In Report of meeting of the interdepart-\
mental committee on forest spraying operations,\
8 November 1967, Ottawa, Ontario. Department of\
Forestry and Rural Development, Ottawa, Ontario,\
Canada. Appendix 1, pp. 1–8.\
Randall, A.P. 1968. Summary report on insecticide\
investigations 1968. Project (CC 1-1) aerial application\
of pesticides for the control of spruce budworm\
C. fumiferana (Clem.) in New Brunswick, 1968. In\
Report of meeting of the interdepartmental committee\
on forest spraying operations, 20–21 November 1968,\
Ottawa, Ontario. Department of Fisheries and Forestry,\
Ottawa, Ontario, Canada. Appendix 5.\
Randall, A.P. 1975. Application technology. In Aerial\
control of forest insects in Canada. Edited by M.L.\
Prebble. Environment Canada, Canadian Forestry\
Service, Ottawa, Ontario, Canada. Pp. 34–55.\
Randall, A.P. and Armstrong, J.A. 1969. Summary report\
on insecticide investigations 1969. Project (CC-1-1)\
chemical control of the spruce budworm C. fumiferana\
(Clem.) in New Brunswick, 1969. In Report of meeting\
of the interdepartmental committee on forest spraying\
operations, 25 November 1969, Ottawa, Ontario.\
Department of Fisheries and Forestry Canadian Forest\
Service, Ottawa, Ontario, Canada. Appendix 7.\
Randall, A.P., Hopewell, W.W., Haliburton, W., and\
Zylstra, B. 1970. Field evaluation of the effectiveness\
of ultra-low-volume application of insecticidal sprays\
from aircraft for the control of the spruce budworm\
(Choristoneura fumiferana Clem) in New Brunswick\
1970. In Report of meeting of the interdepartmental\
committee on forest spraying operations, 29 October\
1970, Ottawa, Ontario. Department of Fisheries and\
Forestry Canadian Forestry Service, Ottawa, Ontario,\
Canada. Appendix 10.\
Randall, A.P., Hopewell, W.W., Haliburton, W., and\
Zylstra, B. 1971. Studies on the control of the\
spruce budworm Choristoneura fumiferana (Clem.) by\
aerial application of the chemicals. In Report of meeting\
of the interdepartmental committee on forest spraying\
operations, 22 November 1971, Ottawa, Ontario.\
Department of the Environment Canadian Forestry\
Service, Ottawa, Ontario, Canada. Appendix 18.\
Randall, A.P. and Nigam, P.C. 1967. Laboratory evalua-\
tion of new insecticidal compounds against spruce\
budworm, larch sawflies, and jackpine sawflies –1966.\
In Report of meeting of the interdepartmental commit-\
tee on forest spraying operations, 3 January 1967,\
Ottawa, Ontario. Department of Forestry and Rural\
Development, Ottawa, Ontario, Canada. Appendix 2.\
Holmes and MacQuarrie S293\
© 2016 Her Majesty the Queen in Right of Canada as represented by\
Natural Resources Canadahttps://doi.org/10.4039/tce.2015.71 Published online by Cambridge University Press\
Régnière, J., Cadogan, L., and Retnakaran, A. 2005.\
Mimic applied against L1 and L5 spruce budworm:\
Manitoba trials 2000/2001 [online]. Spray Efficacy\
Research Group Report. Available from http://www.\
serginternational.org/orderreports.html [accessed\
18 February 2015].\
Retnakaran, A., Gelbic, I., Sundaram, M., Tomkins, W.,\
Ladd, T., Primavera, M., et al. 2001. Mode of action\
of the ecdysone agonist tebufenozide (RH-5992),\
and an exclusion mechanism to explain resistance\
to it. Pest Management Science, 57: 951–957. doi:\
10.1002/ps.377.\
Retnakaran, A., Krell, P., Feng, Q., and Arif, B.M.\
2003. Ecdysone agonists: mechanism and impor-\
tance in controlling insect pests of agriculture\
and forestry. 2003. Archives of Insect Biochemistry\
and Physiology, 54: 187–199. doi: 10.1002/\
arch.10116.\
Retnakaran, A., Smith, L.F.R., Tomkins, W.L.,\
Primavera, M.J., Palli, S.R., and Payne, N.J. 1997.\
Effect of RH-5992, a nonsteroidal ecdysone\
agonist, on the spruce budworm, Choristoneura\
fumiferana (Lepidoptera: Tortricidae): Laboratory,\
greenhouse, and ground spray trials. 1997. The\
Canadian Entomologist, 129: 871–885. doi:\
10.4039/Ent129871-5.\
Reynolds, H.T., Stern, V.M., Fukuto, T.R., and\
Peterson, G.D. 1960. Potential use of Dylox and\
other insecticides in a control program for field crop\
pests in California. Journal of Economic Entomology,\
53: 72 –78. doi:10.1093/jee/53.1.72.\
Richmond, M.L., Henny, C.J., Floyd, R.L., Mannan, R.W.,\
Finch, D.M., and DeWeese, L.R. 1979. Effects of\
Sevin-4-Oil, Dimilin, and Orthene on forest birds.\
Research Paper PSW-148. United States Department\
of Agriculture Forest Service, Pacific Southwest\
Forest and Range Experiment Station, Berkeley,\
California, United States of America.\
Rondeau, G., Sanchez-Bayo, F., Tennekes, H.A.,\
Decourtye, A., Ramirez-Romero, R., and Desneux, N.\
2014. Delayed and time-cumulative toxicity of\
impadlocprid in bees, ants and termites. Scientific\
Reports, 4: 1 –8. doi: 10.1038/srep05566.\
Schmeltz, I. 1971. Nicotine and other tobacco alkaloids.\
In Naturally occurring insecticides. Edited by\
M. Jacobson and D.G. Crosby. Marcel Dekker,\
Inc., New York, New York, United States of\
America. Pp. 99–136.\
Schmutterer, H. 1990. Properties and potential\
of natural pesticides from the neem tree,\
Azadirachta indica. Annual Review of Entomology,\
35: 271–297. doi: 10.1146/annurev.en.35.010190.\
001415.\
Shea, P.J. and Nigam, P.C. 1984. Chemical control. In\
Spruce budworms handbook: managing the spruce\
budworm in western North America. Edited by\
M. Schmitt, D.G. Grimble, and J.L. Searcy. Agriculture\
Handbook 620. United States Department of\
Agriculture, Forest Service, Cooperative State\
Research Service, Washington, District of Columbia,\
United States of America. Pp. 115–132.\
Shore, T.L. and McLean, J.A. 1995. Ambrosia beetles.\
In Forest insect pests in Canada. Edited by J.A.\
Armstrong and W.G.H. Ives. Natural Resources\
Canada Canadian Forest Service, Ottawa, Ontario,\
Canada. Pp. 165–170.\
Sibley, P.K., Kaushik, N.K., and Kreutzweiser, D.P.\
1991. Impact of a pulse application of permethrin on\
the macroinvertebrate community of a headwater\
stream. Environmental Pollution, 70: 35 –55. doi:\
10.1016/0269-7491(91)90130-O.\
Sippell, W.L. and Howse, G.M. 1975. Jack-pine\
budworm. Ontario control projects, 1968–1972. In\
Aerial control of forest insects in Canada. Edited by\
M.L. Prebble. Environment Canada Canadian Forestry\
Service, Ottawa, Ontario, Canada. Pp. 155–156.\
Slaney, G.L., Lantz, V.A., and MacLean, D.A. 2009.\
The economics of carbon sequestration through pest\
management: application to forested landbases in\
New Brunswick and Saskatchewan, Canada. Forest\
Policy and Economics, 11: 525–534. doi: 10.1016/j.\
forpol.2009.07.009.\
Slaney, G.L., Lantz, V.A., and MacLean, D.A. 2010.\
Assessing costs and benefits of pest management on\
forested landbases in eastern and western Canada.\
Journal of Forest Economics, 16: 19 –34. doi:\
10.1016/j.jfe.2009.05.002.\
Smagghe, G. and Degheele, D. 1994. Action of a novel\
nonsteroidal ecdysteroid mimic, tebufenozide\
(RH-5992), on insects of different orders. Pesticide\
Science, 42: 85 –92. doi: 10.1002/ps.2780420204.\
Spear, R.J. 2005. The great gypsy moth war: the\
history of the first campaign in Massachusetts to\
eradicate the gypsy moth, 1890–1901. University\
of Massachusetts Press, Amherst, Massachusetts,\
United States of America.\
Sundaram, K.M.S. 1994. Degradation kinetics of\
tebufenozide in model aquatic systems under con-\
trolled laboratory conditions. Journal of Environ-\
mental Science and Health B, 29: 1081–1104. doi:\
10.1080/03601239409372917.\
Sundaram, K.M.S. 1995. Persistence and fate of\
tebufenozide (RH-5992) insecticide in terrestrial\
microcosms of a forest environment following spray\
application of two Mimic® formulations. Journal of\
Environmental Science and Health B, 30: 321–358.\
doi: 10.1080/03601239509372942.\
Sundaram, K.M.S. 1997a. Persistence and mobility of\
tebufenozide in forest litter and soil ecosystems\
under field and laboratory conditions. Pesticide\
Science, 51: 115–130. doi: 10.1002/(SICI)1096-\
9063(199710)51:2<115::AID-PS599>3.0.CO;2-W.\
Sundaram, K.M.S. 1997b. Persistence of tebufenozide\
in aquatic ecosystems under laboratory and field\
conditions. Pesticide Science, 51: 7 –20. doi: 10.1002/\
(SICI)1096-9063(199709)51:1<7::AID-PS589>\
3.0.CO;2-A.\
Sundaram, K.M.S., Boyonoski, N., and Feng, C.C. 1987.\
Degradation and metabolism of mexacarbate in two\
types of forest litters under laboratory conditions.\
Journal of Environmental Science and Health B, 22:\
29–54. doi: 10.1080/03601238509372544.\
S294 Can. Entomol. Vol. 148, 2016\
© 2016 Her Majesty the Queen in Right of Canada as represented by\
Natural Resources Canadahttps://doi.org/10.4039/tce.2015.71 Published online by Cambridge University Press\
"

# set up openai api
openai_key = "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"
openai.api_key = openai_key


chunk = build_chunk_group(relevance_prompt, text, max_context_length=400, just_one_chunk=True)

print(chunk)
print(get_tokenized_length(chunk[0][0] + chunk[0][1], 'gpt-3.5-turbo'))

response = get_chatgpt_response(relevance_prompt, chunk[0][1])

print(response)
print(yes_or_no(response))
