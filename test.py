import openai
import tiktoken
import time

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

openai_key = "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"
openai.api_key = openai_key

system_message_check_multiple_locations = "You are a yes-or-no machine. This means that you only output 'yes' or 'no', and nothing else. You are given a string and you must determine if the string represents locations from more than one state/province. Answer with 'yes' if more than one state/province is represented and 'no' if only one state/province is represented. Sometimes locations are succeeded by the province and country they inhabit, like 'lac seul area, northwestern ontario, canada', and it is still located in only one province. It is of the utmost importance that you print 'yes' or 'no' and nothing else. Do not say anything other than returning the answer. Do not respond like a human."
term = "zone b (quebec, ontario, manitoba)"

few_shot_examples = [
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

response = get_chatgpt_response(system_message_check_multiple_locations, term, examples=few_shot_examples)

print(response)