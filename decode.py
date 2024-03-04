import json

def parse_obj(obj):
    if isinstance(obj, str):
        return obj.encode('latin_1').decode('utf-8')

    if isinstance(obj, list):
        return [parse_obj(o) for o in obj]

    if isinstance(obj, dict):
        return {key: parse_obj(item) for key, item in obj.items()}

    return obj




with open("message_2.json", "r") as f:
    file_contents = f.read()

decoded_data = parse_obj(json.loads(file_contents))
print(decoded_data)

reversed_data = {}
for key, value in decoded_data.items():
    if isinstance(value, bool):
        # Handle boolean values differently (e.g., keep them unchanged)
        reversed_data[key] = value
    else:
        # Reverse the value (assuming it's a sequence, like a string)
        reversed_data[key] = value[::-1]


with open('decoded_data.json', 'w') as f:
    json.dump(reversed_data, f, ensure_ascii=False, indent=4)  # Use indent for pretty formatting

print("Decoded data has been written to 'decoded_message.json' file.")

