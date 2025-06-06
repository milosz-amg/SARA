from utils.openai import call_openai, update_excel_file


question = "Kim jest Patryk Żywica?"
response = call_openai(question, use_web_search=True)
print(response)

if 'error' not in response:
    if 'response' in response:
        update_excel_file(question, response['response'])
    elif 'results' in response:
        update_excel_file(question, response['results'][0])
else:
    print("Error occurred during OpenAI API call:", response['error'])