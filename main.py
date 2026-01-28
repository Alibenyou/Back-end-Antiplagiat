import os
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = FastAPI()

# 1. Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configuration Supabase (Initialisation UNIQUE et ROBUSTE)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# On définit les options de timeout pour éviter les déconnexions
opts = ClientOptions(
    postgrest_client_timeout=60, # 1 minute
    storage_client_timeout=60
)

# UNE SEULE INITIALISATION ICI
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=opts)

# 3. Chargement du modèle NLP
print("Chargement de l'IA (Hugging Face)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Système prêt !")

# --- Fonctions Utilitaires ---

def extract_text(file_content):
    text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Erreur extraction PDF : {e}")
    return text

def search_google(query):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": 5}
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=10)
        return response.json().get('organic', [])
    except Exception as e:
        print(f"Erreur recherche Google : {e}")
        return []

def scrape_website(url):
    try:
        res = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return " ".join(paragraphs)
    except:
        return ""

# --- La Logique d'Analyse ---

async def process_analysis(analysis_id: str, file_path: str):
    try:
        # On tente de retélécharger si échec
        file_bin = supabase.storage.from_('fichiers_plagiat').download(file_path)
        user_text = extract_text(file_bin)
        
        if not user_text.strip():
            supabase.table("analyses").update({"status": "termine", "plagiarism_score": 0}).eq("id", analysis_id).execute()
            return

        emb_user = model.encode(user_text, convert_to_tensor=True)
        search_query = user_text[:300] 
        web_sources = search_google(search_query)

        max_score = 0
        for source in web_sources:
            link = source['link']
            web_text = scrape_website(link)

            if web_text and len(web_text.strip()) > 100:
                emb_web = model.encode(web_text, convert_to_tensor=True)
                similarity = util.cos_sim(emb_user, emb_web).item()
                score = round(similarity * 100, 2)
                
                if score > max_score:
                    max_score = score

                if score > 15:
                    # Envoi individuel
                    supabase.table("analysis_results").insert({
                        "analysis_id": analysis_id,
                        "url": link,
                        "title": source.get('title'),
                        "similarity_score": score,
                        "matching_text": web_text[:500]
                    }).execute()
        
        # F. Signal de fin
        print(f"✅ Analyse {analysis_id} terminée. Score : {max_score}%")
        supabase.table("analyses").update({
            "status": "termine",
            "plagiarism_score": max_score
        }).eq("id", analysis_id).execute()

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        # En cas d'erreur, on essaie de mettre le statut à 'erreur' pour débloquer l'UI
        try:
            supabase.table("analyses").update({"status": "erreur"}).eq("id", analysis_id).execute()
        except:
            pass

@app.post("/start-analysis/{analysis_id}")
async def start_analysis(analysis_id: str, background_tasks: BackgroundTasks):
    try:
        res = supabase.table("analyses").select("file_path").eq("id", analysis_id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Analyse non trouvée")
            
        file_path = res.data['file_path']
        background_tasks.add_task(process_analysis, analysis_id, file_path)
        return {"message": "Analyse lancée"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # On récupère le port donné par l'hébergeur (Render, Railway, etc.)
    # Si on est en local, on utilise 8000 par défaut
    port = int(os.environ.get("PORT", 8000))
    
    # "0.0.0.0" permet d'écouter toutes les interfaces réseau (indispensable pour Docker)
    uvicorn.run(app, host="0.0.0.0", port=port)
