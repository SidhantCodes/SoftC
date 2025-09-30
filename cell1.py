import pandas as pd
import numpy as np
import os

# --- 1. Setup and Configuration ---
# Define the input file from Kaggle and the output directory
input_filename = "asthma-disease-dataset.csv"
output_dir = "new_asthma_dataset"
os.makedirs(output_dir, exist_ok=True)

# Define symptom templates for each asthma zone
symptom_templates = {
    "green": [
    "Feeling good today, breathing is easy.",
    "No issues with my breathing at all.",
    "Was able to exercise without any problems.",
    "Slept well, no coughing or wheezing.",
    "My breathing feels clear and normal.",
    "No chest tightness or shortness of breath today.",
    "Had a calm and restful night without waking up coughing.",
    "No wheezing or discomfort while climbing stairs.",
    "Breathing feels smooth and effortless.",
    "Enjoyed a morning jog without any trouble.",
    "Able to complete my daily tasks comfortably.",
    "Feeling energetic and without any respiratory discomfort.",
    "No inhaler needed today, feeling normal.",
    "Could take deep breaths without pain or tightness.",
    "Not experiencing any shortness of breath at all.",
    "Walking to work was easy and effortless.",
    "Breathing rate is normal and relaxed.",
    "No coughing after meals or exercise.",
    "Chest feels light and unrestricted.",
    "Slept through the night without asthma interruptions.",
    "No signs of fatigue from respiratory effort.",
    "Feeling completely symptom-free today.",
    "No tightness in the chest this morning.",
    "Breathing feels steady and comfortable.",
    "Able to climb stairs without wheezing.",
    "Airway feels clear and unobstructed.",
    "No difficulty speaking or taking deep breaths.",
    "Feeling completely in the green zone today.",
    "Able to engage in moderate exercise easily.",
    "Breathing feels effortless after stretching.",
    "Feeling strong and healthy in terms of respiration.",
    "No wheezing while doing light household chores.",
    "Resting heart and breathing rates are normal.",
    "Airflow feels unrestricted and calm.",
    "Able to enjoy outdoor activities without issues.",
    "Feeling light, clear-headed, and without chest tightness.",
    "No asthma-related fatigue present today.",
    "Could go for a long walk comfortably.",
    "Feeling normal and symptom-free throughout the day.",
    "Breathing feels relaxed even after minor exertion.",
    "Able to talk in full sentences easily.",
    "No coughing fits during normal activity.",
    "Feeling comfortable and stable in my breathing.",
    "Chest feels relaxed and breathing is deep.",
    "Able to climb hills or stairs without shortness of breath.",
    "No wheezing after minor activity or exertion.",
    "Feeling strong and capable in terms of breathing.",
    "Able to run errands without breathing difficulty.",
    "Airway feels smooth and unrestricted all day.",
    "No discomfort when taking deep breaths.",
    "Feeling fully capable of exercise or movement.",
    "Able to perform work or chores comfortably.",
    "Feeling light and easy-breathing during daily activities.",
    "No need for inhaler use today.",
    "Chest feels open and relaxed after activity.",
    "Able to sleep comfortably and deeply.",
    "No coughing or wheezing throughout the day.",
    "Feeling energetic with normal respiratory function.",
    "Able to enjoy hobbies without respiratory strain.",
    "No signs of asthma while walking or climbing.",
    "Breathing is effortless and controlled.",
    "Feeling normal, calm, and in control of my breathing.",
    "Able to complete physical activity without issue.",
    "Feeling positive and symptom-free this afternoon.",
    "No chest tightness or shortness of breath noticed.",
    "Able to participate in social activities comfortably.",
    "Feeling strong and healthy in breathing today.",
    "No coughing after exertion or activity.",
    "Able to take deep breaths without discomfort.",
    "Feeling light, comfortable, and symptom-free.",
    "Chest feels clear and airways unobstructed.",
    "Breathing is steady and comfortable at rest.",
    "Able to perform daily tasks without extra effort.",
    "No fatigue from respiratory effort this morning.",
    "Feeling energetic and normal in respiratory function.",
    "Able to enjoy exercise or activity easily.",
    "No wheezing or discomfort after physical activity.",
    "Breathing feels calm and effortless all day.",
    "Feeling completely symptom-free and well.",
    "Able to speak without interruption or breathlessness.",
    "No need for rescue inhaler today.",
    "Feeling fully capable of activity and exercise.",
    "Breathing is clear and comfortable at all times.",
    "Able to complete daily work without shortness of breath.",
    "No chest tightness during normal activity.",
    "Feeling comfortable and unrestricted in respiration.",
    "Able to enjoy walking and light jogging easily.",
    "Breathing is deep, calm, and regular.",
    "Feeling well and free from asthma symptoms.",
    "No wheezing noticed during minor exertion.",
    "Able to perform household chores without fatigue.",
    "Chest feels relaxed and open all day.",
    "Feeling positive and symptom-free this morning.",
    "Able to talk easily without taking extra breaths.",
    "Breathing feels clear and unrestricted after activity.",
    "Feeling light and comfortable in respiratory function.",
    "Able to engage in moderate physical activity comfortably.",
    "No signs of asthma during daily routines.",
    "Chest feels normal and breathing effortless.",
    "Feeling energetic, healthy, and symptom-free.",
    "Able to sleep and wake comfortably without coughing.",
    "Breathing is steady and unlabored.",
    "Feeling strong, capable, and in control of breathing.",
    "Able to participate in social or recreational activity.",
    "No shortness of breath after minor exercise.",
    "Feeling normal and calm throughout the day.",
    "Able to enjoy daily activities without respiratory effort.",
    "Breathing feels open and natural.",
    "Feeling completely healthy in terms of respiration.",
    "No wheezing after walking or light activity.",
    "Able to complete errands or tasks comfortably.",
    "Chest feels clear and breathing steady.",
    "Feeling light, positive, and symptom-free today.",
    "Able to participate in physical activity with ease.",
    "No coughing or shortness of breath noted.",
    "Breathing is calm and steady all day.",
    "Feeling energetic and normal without symptoms.",
    "Able to engage in work or hobbies comfortably.",
    "Chest feels relaxed and breathing effortless."
    ],

    "yellow": [
    "Feeling a little short of breath this afternoon.",
    "My chest feels a bit tight this morning.",
    "Woke up coughing a couple of times last night.",
    "Had to use my rescue inhaler once after a walk.",
    "Getting tired more easily than usual.",
    "A bit of wheezing after I laughed hard.",
    "Breathing feels slightly labored after climbing stairs.",
    "Noticing minor tightness in my chest.",
    "Feeling some shortness of breath while walking.",
    "Had to pause briefly after exertion due to breathlessness.",
    "A little wheezing after a mild jog.",
    "Breathing feels heavier than normal today.",
    "Feeling slightly fatigued from normal activity.",
    "Needed to use inhaler once during daily chores.",
    "Chest feels mildly constricted at times.",
    "Slight cough after physical activity.",
    "Breathing rate slightly elevated during walking.",
    "Feeling some discomfort in the chest after climbing stairs.",
    "Had minor wheezing while playing with my kids.",
    "Feeling a little out of breath after light exercise.",
    "Breathing feels okay at rest, but labored during exertion.",
    "Mild tightness noticed when taking deep breaths.",
    "Feeling slightly fatigued by normal activity.",
    "Had to slow down while exercising due to shortness of breath.",
    "Some mild coughing after climbing a few stairs.",
    "Breathing feels slightly uneven or shallow.",
    "Chest feels a bit heavy after walking a short distance.",
    "Feeling mildly tired due to minor breathing discomfort.",
    "Needed to take a few extra breaths after minor activity.",
    "Experiencing minor wheezing intermittently.",
    "Slight chest tightness after household chores.",
    "Feeling a little uncomfortable while breathing.",
    "Breathing feels slightly restricted at times.",
    "Had to rest briefly after a short walk.",
    "Feeling mildly anxious due to minor breathlessness.",
    "Some wheezing noticed after light activity.",
    "Chest feels slightly tense this morning.",
    "Feeling somewhat short of breath during activity.",
    "Breathing feels a bit uneven after climbing stairs.",
    "Had a mild cough after minor exertion.",
    "Chest feels a little tight after stretching.",
    "Feeling slightly off during exercise due to breathing.",
    "Breathing feels heavier than usual during activity.",
    "Some discomfort in chest noticed while walking.",
    "Feeling a bit fatigued after daily tasks.",
    "Needed to pause briefly due to minor breathlessness.",
    "Experiencing mild wheezing during light activity.",
    "Chest feels slightly tight after mild exertion.",
    "Feeling a little winded after climbing stairs.",
    "Breathing feels somewhat labored after activity.",
    "Had minor coughing after walking a short distance.",
    "Feeling slightly uncomfortable with my breathing today.",
    "Chest feels a bit tight at times throughout the day.",
    "Experiencing mild fatigue due to shortness of breath.",
    "Breathing feels a little heavier than normal.",
    "Had to use inhaler once after climbing stairs.",
    "Feeling somewhat restricted while taking deep breaths.",
    "Slight wheezing noticed intermittently today.",
    "Chest feels mildly constricted after minor activity.",
    "Feeling a little tired due to mild asthma symptoms.",
    "Breathing feels slightly shallow during exertion.",
    "Had a small coughing episode after walking.",
    "Feeling slightly uneasy due to minor chest tightness.",
    "Some shortness of breath noticed while moving around.",
    "Chest feels a little tight after light activity.",
    "Feeling slightly fatigued from breathing discomfort.",
    "Breathing feels heavier than usual at times.",
    "Had minor wheezing after household chores.",
    "Feeling a little breathless after climbing stairs.",
    "Chest feels slightly constricted intermittently.",
    "Experiencing mild discomfort while taking deep breaths.",
    "Breathing feels somewhat restricted after mild exertion.",
    "Feeling a little off due to minor respiratory symptoms.",
    "Had to slow down while exercising today.",
    "Chest feels slightly tight after walking a short distance.",
    "Feeling mildly fatigued from minor asthma discomfort.",
    "Breathing feels a little heavy after mild activity.",
    "Some mild wheezing noticed after exertion.",
    "Feeling slightly short of breath intermittently.",
    "Chest feels a little tight while taking deep breaths.",
    "Breathing feels slightly uneven after light activity.",
    "Feeling somewhat fatigued due to mild asthma symptoms.",
    "Had minor coughing after climbing stairs.",
    "Chest feels mildly constricted while resting.",
    "Feeling a little off due to minor breathing difficulty.",
    "Breathing feels heavier than usual after minor activity.",
    "Some mild shortness of breath noticed intermittently.",
    "Chest feels slightly tight during light exertion.",
    "Feeling somewhat fatigued from minor respiratory discomfort.",
    "Breathing feels a little restricted while walking.",
    "Had to pause briefly due to mild shortness of breath.",
    "Chest feels slightly constricted after light activity.",
    "Feeling mildly fatigued from minor exertion today.",
    "Breathing feels slightly heavier than usual.",
    "Some mild wheezing noticed during minor activity.",
    "Feeling slightly short of breath while performing tasks.",
    "Chest feels a little tight intermittently.",
    "Breathing feels a little restricted after minor exertion.",
    "Feeling somewhat fatigued due to mild respiratory symptoms.",
    "Had minor coughing episodes intermittently today.",
    "Chest feels slightly constricted after mild activity.",
    "Feeling a little uneasy due to minor asthma symptoms.",
    "Breathing feels somewhat heavier than normal during activity.",
    "Some mild wheezing noticed intermittently.",
    "Feeling slightly off due to minor chest tightness."
    ],

    "red": [
    "It's very hard to breathe, even while I'm resting.",
    "My rescue inhaler isn't helping much at all.",
    "I can't speak in full sentences without taking a breath.",
    "There's constant wheezing and my chest feels very tight.",
    "Feeling very anxious because I'm struggling for air.",
    "My symptoms are getting worse and I feel terrible.",
    "Breathing feels almost impossible right now.",
    "My chest is extremely tight and painful.",
    "I can't lie down without feeling suffocated.",
    "Air feels heavy and difficult to inhale.",
    "I feel panicked because I can't catch my breath.",
    "Wheezing is constant and getting worse.",
    "Every breath feels like a struggle.",
    "I'm having trouble walking even a few steps.",
    "Chest tightness is severe and persistent.",
    "I need my inhaler urgently, it's not enough.",
    "Breathing feels shallow and labored constantly.",
    "I feel dizzy from lack of oxygen.",
    "Every movement makes breathing more difficult.",
    "My heart feels like it's racing due to breathlessness.",
    "Shortness of breath is severe and unrelenting.",
    "I feel like I'm suffocating at rest.",
    "Chest pain accompanies extreme shortness of breath.",
    "I cannot talk more than a few words without gasping.",
    "Wheezing and coughing are constant and painful.",
    "Feeling extremely weak because of lack of air.",
    "I have to sit upright to breathe at all.",
    "Breathing feels obstructed and very tight.",
    "I feel panicked because my lungs won't expand.",
    "Shortness of breath is overwhelming and frightening.",
    "Chest feels constricted and airways tight.",
    "Every inhalation feels insufficient.",
    "I feel desperate for more air.",
    "My inhaler provides little relief.",
    "Breathing requires intense effort constantly.",
    "Feeling exhausted due to severe breathlessness.",
    "I cannot perform even light activity without gasping.",
    "Chest tightness is severe enough to be alarming.",
    "I feel trapped due to my inability to breathe.",
    "Air feels thick and hard to pull in.",
    "My symptoms are worsening rapidly.",
    "I feel extremely weak from lack of oxygen.",
    "Every breath is accompanied by painful wheezing.",
    "I cannot lie down or sleep due to breathlessness.",
    "Feeling extremely anxious and panicked about breathing.",
    "Chest feels extremely tight, like a heavy weight.",
    "Wheezing is continuous and alarming.",
    "Shortness of breath prevents normal conversation.",
    "I feel very lightheaded due to low oxygen.",
    "Breathing is labored even at rest.",
    "My rescue inhaler is barely effective.",
    "Chest feels constricted and airways tight.",
    "I feel terrified because I cannot breathe properly.",
    "Shortness of breath is severe and constant.",
    "Every movement worsens my breathing.",
    "I cannot take a full deep breath at all.",
    "Breathing feels obstructed and very painful.",
    "I feel desperate for air constantly.",
    "Chest tightness is severe and alarming.",
    "I cannot talk without gasping for air.",
    "Breathing requires maximum effort continually.",
    "Feeling extremely panicked due to severe asthma symptoms.",
    "Air feels difficult to inhale fully.",
    "Chest feels like it is under extreme pressure.",
    "Shortness of breath is overwhelming.",
    "Wheezing is loud, constant, and distressing.",
    "I cannot rest because breathing is too difficult.",
    "Feeling extremely weak and dizzy from lack of oxygen.",
    "Breathing is almost impossible without support.",
    "My symptoms feel life-threatening.",
    "Chest feels extremely heavy and tight.",
    "I cannot move without feeling breathless.",
    "Every breath feels shallow and insufficient.",
    "Feeling terrified because inhaler is not working.",
    "Wheezing is intense and persistent.",
    "Shortness of breath prevents normal activity.",
    "I feel panic because I can't get enough air.",
    "Chest tightness is severe and constant.",
    "Breathing requires extreme effort continually.",
    "Feeling extremely anxious and breathless.",
    "Air feels hard to pull into my lungs.",
    "Chest feels constricted and painful.",
    "Shortness of breath is severe and frightening.",
    "Wheezing and coughing are constant.",
    "I feel weak and dizzy due to low oxygen.",
    "Every movement makes breathing worse.",
    "Cannot speak more than a few words without gasping.",
    "Chest tightness and pain are overwhelming.",
    "I feel suffocated and panicked.",
    "Breathing feels almost impossible at all times.",
    "My inhaler barely helps.",
    "Feeling extremely exhausted from lack of air.",
    "Chest feels extremely tight and heavy.",
    "Shortness of breath is severe and ongoing.",
    "Wheezing is constant and loud.",
    "Cannot perform even light activities without gasping.",
    "Feeling terrified and panicked from lack of oxygen.",
    "Air feels thick and hard to inhale.",
    "Chest feels extremely constricted.",
    "Breathing feels insufficient and painful.",
    "I cannot rest or sleep due to breathlessness.",
    "Feeling weak, dizzy, and frightened.",
    "Shortness of breath feels overwhelming.",
    "Chest pain accompanies severe breathing difficulty.",
    "Every breath feels like a struggle.",
    "Feeling desperate for air constantly.",
    "Wheezing and tightness are severe.",
    "Cannot talk more than a few words without stopping.",
    "Breathing requires maximum effort continually.",
    "Feeling extremely anxious, weak, and breathless.",
    "Air feels very hard to pull in.",
    "Chest feels heavy, tight, and constricted.",
    "Shortness of breath is severe and frightening.",
    "Wheezing and coughing are constant and severe.",
    "Feeling panicked, weak, and struggling to breathe.",
    "Cannot move without extreme shortness of breath.",
    "Chest tightness and pain are overwhelming.",
    "Breathing feels almost impossible at all times.",
    "My inhaler is not helping sufficiently.",
    "Feeling extremely exhausted and anxious from lack of oxygen.",
    "Shortness of breath prevents all activity.",
    "Chest feels extremely tight and painful.",
    "Wheezing is continuous and severe."
    ]
}


# --- 2. Define Helper Functions ---

def generate_pef_reading(row, personal_best):
    """
    Intelligently generates a PEF reading based on symptom flags.
    If symptoms are present, it's more likely to be a lower PEF.
    """
    # Count how many key symptom flags are active for this patient
    symptom_score = row['Wheezing'] + row['Shortness of Breath'] + row['Chest Tightness']
    
    # Define PEF percentage ranges based on symptom score
    if symptom_score >= 2:  # High chance of being a Red Zone event
        # Simulate PEF between 30% and 60% of personal best
        pef_percentage = np.random.uniform(0.30, 0.60)
    elif symptom_score == 1:  # High chance of being a Yellow Zone event
        # Simulate PEF between 55% and 85% of personal best
        pef_percentage = np.random.uniform(0.55, 0.85)
    else:  # No symptoms, high chance of being a Green Zone event
        # Simulate PEF between 80% and 105% of personal best
        pef_percentage = np.random.uniform(0.80, 1.05)
        
    return int(personal_best * pef_percentage)

def classify_asthma_zone(pef_reading, personal_best):
    """Classifies the PEF reading into Green, Yellow, or Red zones."""
    percentage = (pef_reading / personal_best) * 100
    if percentage >= 80:
        return "green"
    elif 50 <= percentage < 80:
        return "yellow"
    else:
        return "red"

def generate_symptom_text(zone):
    """Returns a random symptom report based on the zone."""
    return np.random.choice(symptom_templates[zone])

# --- 3. Main Data Processing Logic ---

try:
    # Load the base dataset from Kaggle
    df = pd.read_csv(input_filename)
    print(f"Successfully loaded '{input_filename}' with {len(df)} rows.")

    # Step 1: Simulate a "personal best" PEF for each patient
    # A simple but effective simulation based on a reasonable range
    df['personal_best_pef'] = np.random.randint(400, 550, df.shape[0])

    # Step 2: Intelligently generate the current PEF reading for each row
    df['pef_reading'] = df.apply(lambda row: generate_pef_reading(row, row['personal_best_pef']), axis=1)

    # Step 3: Define the flare state based on the new PEF reading
    df['flare_state'] = df.apply(lambda row: classify_asthma_zone(row['pef_reading'], row['personal_best_pef']), axis=1)

    # Step 4: Synthesize a text report based on the flare state
    df['report_text'] = df['flare_state'].apply(generate_symptom_text)

    # --- 4. Save the Enriched Dataset ---
    output_filename = os.path.join(output_dir, "processed_asthma_dataset.csv")
    df.to_csv(output_filename, index=False)

    print(f"\nData synthesis complete. Enriched data saved to '{output_filename}'")
    
    # Display the new columns for verification
    print("\n--- Sample of the new, enriched data ---")
    print(df[['Wheezing', 'Shortness of Breath', 'personal_best_pef', 'pef_reading', 'flare_state', 'report_text']].head(10))

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please download the 'Asthma Disease Dataset' from Kaggle and place it in the same directory as this script.")

