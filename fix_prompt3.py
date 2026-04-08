with open("src/hf_gaia_agent/graph.py", "r", encoding="utf-8") as f:
    text = f.read()

old_str = "For Wikipedia subjects (people, organizations, events), ALWAYS prefer search_wikipedia and fetch_wikipedia_page over generic web_search."
new_str = "For Wikipedia subjects, prefer search_wikipedia and fetch_wikipedia_page. However, fetch_wikipedia_page omits tables and lists! If you are looking for tables of data (like rosters, awards, winner lists), you MUST use extract_tables_from_url or fetch_url on the Wikipedia page URL instead of fetch_wikipedia_page."

text = text.replace(old_str, new_str)
with open("src/hf_gaia_agent/graph.py", "w", encoding="utf-8") as f:
    f.write(text)
