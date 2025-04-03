from groq import Groq

client = Groq(
    api_key= '<GROQ API KEY>',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "WAP to generate a star with triangle ",
        }
    ],
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)


