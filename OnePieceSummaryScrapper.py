import requests
from bs4 import BeautifulSoup

# Generate a list of URLs for the first 1100 OP chapter listed on onepiece fandom
base_url = "https://onepiece.fandom.com/wiki/Chapter_{}"
urls = [base_url.format(i) for i in range(1, 1100)]
special_characters = {
    '`': "'",
    '´': "'",
    '°C': '',
    '\xa0': ' ',
    'ß': 'ss',
    'à': 'a',
    'è': 'e',
    'é': 'e',
    'ê': 'e',
    'É': 'E',
    'ï': 'i',
    'û': 'u',
    'ö': 'o',
    'ô': 'o',
    'ō': 'oo',
    'Ō': 'Oo',
    'œ': 'oe',
    'ū': 'u',
    '–': '-',
    '—': '-',
    '―': '-',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '…': '...',
    ' [ゴゴゴゴ]': '',
    ' "女難"': '',
    '\ufeff': ''
}

# Function to get and extract the desired section from a URL
def scrape_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            start_header = soup.find('span', id='Long_Summary')
            if start_header:
                next_header = start_header.find_next('h2')
                content = ''
                current_tag = start_header
                while current_tag and current_tag != next_header:
                    if current_tag.name == 'p':
                        content += current_tag.get_text()
                    current_tag = current_tag.find_next()
                return content
            else:
                # Episode 314 markdown issue
                start_header = soup.find('span', id='Summary')
                if start_header:
                    next_header = start_header.find_next('h2')
                    content = ''
                    current_tag = start_header
                    while current_tag and current_tag != next_header:
                        if current_tag.name == 'p':
                            content += current_tag.get_text()
                        current_tag = current_tag.find_next()
                    return content
                else:
                    print(f"Section not found on {url}")
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Iterate over URLs, scrape, and concatenate the sections
concatenated_sections = ''
for url in urls:
    section = scrape_url(url)
    if section:
        concatenated_sections += section + '\n'

# Replace special characters to 'simplify' the text
for old, new in special_characters.items():
    concatenated_sections = concatenated_sections.replace(old, new)

# Write the concatenated sections to a file
with open("utils/OnePieceSummary.txt", 'w', encoding='utf-8') as f:
    f.write(concatenated_sections)

print("Text theft completed.")