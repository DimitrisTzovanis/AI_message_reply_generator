import json
import re
import pandas as pd
import unicodedata
import os

def parse_obj(obj):
    if isinstance(obj, str):
        return obj.encode('latin_1').decode('utf-8')

    if isinstance(obj, list):
        return [parse_obj(o) for o in obj]

    if isinstance(obj, dict):
        return {key: parse_obj(item) for key, item in obj.items()}

    return obj


def is_number(s):
    try:
        float(s)  # Attempt to convert to float
        return True
    except ValueError:
        return False
    
def extract_substring(s):
    # Take the first 10 characters of the string
    substr = s[:10]
    
    # Check if the last character is not a space and the substring is not the whole string
    if substr[-1] != ' ' and len(s) > 10:
        # Find the next space in the string after the 10th character
        next_space = s.find(' ', 10)
        
        # If there is a next space, extend the substring to the next space
        # If there isn't a next space, take the whole string
        substr = s[:next_space] if next_space != -1 else s
    
    return substr

def is_specific_character(s):
    return s in [':', '!']



def extract_content(messagesPrev, sender_name1, sender_name2):
    messages = messagesPrev["messages"]
    contents1 = []
    contents2 = []
    for i in range(len(messages) - 1):
<<<<<<< HEAD
         if messages[i]['sender_name'] == "Person1" and messages[i + 1]['sender_name'] == "Elpida Stasinou":
            if 'content' in messages[i] and 'content' in messages[i+1] and len(messages[i])<40 and len(messages[i+1]['content']) < 25 and 'http' not in messages[i]['content'] and 'http' not in messages[i+1]['content'] and 'profit bird' not in messages[i]['content'] and 'profit bird' not in messages[i+1]['content']:
                flag = False
                rr = messages[i]['content']
                rr = ''.join(c for c in unicodedata.normalize('NFD', rr) if unicodedata.category(c) != 'Mn')
                rr = re.sub(r'[,"\'\']', '', rr)
                rr = re.sub(r'[^\w\s,;.?!:Ά-ώ]+', '', rr)
=======
         if messages[i]['sender_name'] == "Person1" and messages[i + 1]['sender_name'] == "Person2":
            if 'content' in messages[i] and 'content' in messages[i+1] and 'http' not in messages[i]['content'] and 'http' not in messages[i+1]['content'] and 'profit bird' not in messages[i]['content'] and 'profit bird' not in messages[i+1]['content']:
                contents1.append(messages[i]['content'])
>>>>>>> 8137a53c49a68dc643b6db210b8119a97bfe7eae
                reply = messages[i + 1]['content']
                k=i+2
                n=0
                while(messages[k]['sender_name'] == "Person2") and n<2:
                    if 'content' in messages[k] and len(messages[k]['content']) < 25 and 'http' not in messages[k]['content'] and 'profit bird' not in messages[k]['content'] and 'video chat' not in messages[k]['content']:
                        reply+= " "
                        reply+= messages[k]['content']
                    k+=1
                    n+=1
                reply = re.sub(r'[,"\'\']', '', reply)
                reply = re.sub(r'[^\w\s,;.?!:Ά-ώ]+', '', reply)
                reply = ''.join(c for c in unicodedata.normalize('NFD', reply) if unicodedata.category(c) != 'Mn')
                
                if not reply or len(rr) < 3 or len(rr) > 40 or not rr or is_number(rr) or is_specific_character(rr) or is_number(reply) or is_specific_character(reply):
                    flag = True
                if flag == False:
                    contents1.append(rr)
                    contents2.append(reply)
    return contents1, contents2



 
directory_path = '/Users/dimitris/Desktop/v1/messages'  # Update this to your folder's path

# List all files in the directory
all_files = os.listdir(directory_path)

# Filter for JSON files
json_files = [file for file in all_files if file.endswith('.json')]

# Initialize a list to hold the parsed data
sc1 = []
sc2 = []

# Loop through each JSON file and parse it
for json_file in json_files:
    with open(os.path.join(directory_path, json_file), 'r') as f:
        file_contents = f.read()
    
    # Assuming the JSON structure matches what parse_obj expects
    decoded_data = parse_obj(json.loads(file_contents))

    reversed_data = {}
    for key, value in decoded_data.items():
        if isinstance(value, bool):
            # Handle boolean values differently (e.g., keep them unchanged)
            reversed_data[key] = value
        else:
            # Reverse the value (assuming it's a sequence, like a string)
            reversed_data[key] = value[::-1]

    sender_name1 = "Person1"
    sender_name2 = "Person2"

    decoded_data =  json.dumps(reversed_data)

    sender_contents1, sender_contents2 = extract_content(json.loads(decoded_data), sender_name1, sender_name2)
    sc1 += sender_contents1
    sc2 += sender_contents2
    
paired_messages = zip(sc1, sc2)
print("Decoded data has been written to file.")
# Create a DataFrame from the paired messages
df = pd.DataFrame(paired_messages, columns=['user', 'response'])

# Save the DataFrame to a CSV file
df.to_csv('conversations.csv', index=False, encoding='utf-8')

