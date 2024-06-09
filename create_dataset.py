import json


def extract_content(messagesPrev, sender_name1, sender_name2):
    messages = messagesPrev["messages"]
    contents1 = []
    contents2 = []
    contentsCombined = []
    for i in range(len(messages) - 1):
         if messages[i]['sender_name'] == "Person1" and messages[i + 1]['sender_name'] == "Person2":
            if 'content' in messages[i] and 'content' in messages[i+1] and 'http' not in messages[i]['content'] and 'http' not in messages[i+1]['content'] and 'profit bird' not in messages[i]['content'] and 'profit bird' not in messages[i+1]['content']:
                contents1.append(messages[i]['content'])
                reply = messages[i + 1]['content']
                k=i+2
                while(messages[k]['sender_name'] == "Elpida Stasinou"):
                    if 'content' in messages[k] and 'http' not in messages[k]['content'] and 'profit bird' not in messages[k]['content']:
                        reply+= " "
                        reply+= messages[k]['content']
                    k+=1
                contents2.append(reply)
                contentsCombined.append(messages[i]['content']+" "+reply)
    return contents1, contents2, contentsCombined


    
with open("decoded_data.json", "r") as f:
    file_contents = f.read()

sender_name1 = "Δημητρης Τζοβανης"
sender_name2 = "Elpida Stasinou"

sender_contents1, sender_contents2, combined = extract_content(json.loads(file_contents), sender_name1, sender_name2)

parsed_data = json.loads(file_contents)

# Access sender_name and content from each message

with open('person1.txt', 'w') as f:
    json.dump(sender_contents1, f, ensure_ascii=False, indent=4)  # Use indent for pretty formatting

with open('person2.txt', 'w') as f:
    json.dump(sender_contents2, f, ensure_ascii=False, indent=4)  # Use indent for pretty formatting

with open('combined.txt', 'w') as f:
    json.dump(combined, f, ensure_ascii=False, indent=4)  # Use indent for pretty formatting

try:
    with open('combined.txt', 'r') as file:
        content = file.read()
    # Perform the operation of removing all commas and quotation marks
    cleaned_content = content.replace(',', '').replace('"', '')

    # Use cleaned_content with json.dump
    with open('combined.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
except Exception as e:
    print(f"An error occurred: {e}")

print("Decoded data has been written to 'decoded_message.json' file.")

