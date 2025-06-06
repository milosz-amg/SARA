from utils.openai import call_openai, update_excel_file


question = "Podaj informacje o grantach badawczych dostępnych dla projektów AI w medycynie w 2024 roku"
response = call_openai(question, use_web_search=True)

if 'error' not in response:
    if 'response' in response:
        update_excel_file(question, response['response'])
    elif 'results' in response:
        update_excel_file(question, response['results'][0])
else:
    print("Error occurred during OpenAI API call:", response['error'])