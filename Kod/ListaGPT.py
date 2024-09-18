import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta
import io

# Initialize variables
next_call_time = datetime.now()
rate_limit = 0.15
lock = asyncio.Lock()

async def fetch_openai_response(session, message, model, api_key):
    global next_call_time
    async with lock:
        await asyncio.sleep(max((next_call_time - datetime.now()).total_seconds(), 0))
        next_call_time = datetime.now() + timedelta(seconds=rate_limit)
    try:
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        body = {
            "model": model,
            "messages": message,
            "temperature": 0
        }

        async with session.post(url, json=body, headers=headers) as response:
            response_data = await response.json()
            if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return "Error: Invalid response data format or empty choices."
    except Exception as e:
        return f"An error occurred: {e}"

async def process_dataframe_row(session, row, index, num_rows, column_name, prompt, model, api_key, status_text):
    # Uppdatera texten för att visa vilken rad som bearbetas
    status_text.text(f"Bearbetar rad {index + 1} av {num_rows}")
    
    text = row[column_name]

    if pd.isna(text) or str(text).strip() == '':
        return None
    text = str(text).strip()
    messages = [
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": text}
    ]
    
    response = await fetch_openai_response(session, messages, model, api_key)
    if "Error:" in response:
        return None 
    else:
        return response

async def process_comments(df, column_name, prompt, model, api_key, num_rows, max_concurrent_requests=20):
    st.write("Startar bearbetning...")
    start_time = datetime.now()
    if num_rows is None:
        num_rows = len(df)

    # Initiera progressbar och status-text
    progress_bar = st.progress(0)
    status_text = st.empty()  # Skapa en tom plats för statusuppdateringar
    total_tasks = min(num_rows, len(df))

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            process_dataframe_row(session, row, index, num_rows, column_name, prompt, model, api_key, status_text)
            for index, row in df.head(num_rows).iterrows()
        ]
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append((result, i))
            progress_bar.progress((i + 1) / total_tasks)
       
    # Sortera resultaten efter ursprunglig ordning
    results.sort(key=lambda x: x[1])
    processed_results = [res[0] for res in results]

    # Uppdatera DataFrame med resultaten
    for result, (index, _) in zip(processed_results, df.head(num_rows).iterrows()):
        df.at[index, 'Resultat'] = result if result is not None else "Timeout"
    
    end_time = datetime.now()
    st.write(f"Bearbetning slutförd på: {end_time - start_time}")
    return df

def main():
    st.title("Textanalys med OpenAI API")

    # Mata in OpenAI API-nyckel
    api_key = st.text_input("Ange din OpenAI API-nyckel", type="password")

    # Ladda upp fil
    uploaded_file = st.file_uploader("Ladda upp en Excel-fil", type=["xlsx"])

    if uploaded_file is not None and api_key:
        df = pd.read_excel(uploaded_file)

        # Välj kolumn att analysera
        column_name = st.selectbox("Välj kolumn att analysera", df.columns)

        # Välj analys typ
        analysis_type = st.selectbox("Välj typ av analys", ["Sentimentanalys", "Summering", "Kategoriindelning"])

        # Definiera prompts för olika analys typer
        prompts = {
            "Sentimentanalys": """
Bedöm sentimentet i följande text. Svara ENDAST med:
"Positiv"
"Negativ"
"Neutral"
""",
            "Summering": """
Sammanfatta följande text i en mening.
""",
            "Kategoriindelning": """
Bestäm vilken kategori följande text tillhör. Välj mellan:
"Nyheter"
"Sport"
"Underhållning"
"Teknik"
"""
        }

        # Hämta vald prompt
        prompt = prompts[analysis_type]

        # Visa prompten för användaren
        st.write("Använd följande prompt för analysen:")
        st.code(prompt, language="markdown")

        # Ge användaren möjlighet att redigera prompten
        prompt = st.text_area("Du kan redigera prompten här om du vill:", value=prompt, height=150)

        # Välj OpenAI-modell
        model = st.selectbox("Välj OpenAI-modell", ["gpt-4o-mini","gpt-3.5-turbo", "gpt-4"])

        # Antal rader att bearbeta
        num_rows = st.number_input(
            "Ange antal rader att bearbeta (lämna standardvärdet för alla)", 
            min_value=1, 
            max_value=len(df), 
            value=len(df)
        )

        # Ange filnamn för nedladdning
        filename = st.text_input("Ange filnamn för nedladdad fil", value="bearbetad_data.xlsx")

        if st.button("Starta bearbetning"):
            # Kör bearbetningen
            df_processed = asyncio.run(
                process_comments(df, column_name, prompt, model, api_key, num_rows)
            )

            # Förbered DataFrame för nedladdning
            towrite = io.BytesIO()
            df_processed.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Ladda ner bearbetad data",
                data=towrite,
                file_name=filename,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

if __name__ == '__main__':
    main()