import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_medical_page(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Example: extract title and paragraphs – adjust selectors based on the site
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'Unknown'
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    content = "\n".join(paragraphs)
    return {'title': title, 'content': content}

def main():
    # Example URLs; replace with your list of target medical sites
    urls = [
        'https://example-medical-site.com/disease-info',
        # Add more URLs here
    ]
    data = []
    for url in urls:
        try:
            print(f"Scraping {url}")
            data.append(scrape_medical_page(url))
            time.sleep(1)  # polite delay
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    df = pd.DataFrame(data)
    df.to_csv('data/scraped_data.csv', index=False)
    print("Scraping complete. Data saved to data/scraped_data.csv.")

if __name__ == "__main__":
    main()
