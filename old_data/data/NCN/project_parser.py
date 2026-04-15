import json
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager

BASE_URL = "https://projekty.ncn.gov.pl"
START_URL = "https://projekty.ncn.gov.pl/index.php?jednostka=Uniwersytet+Adama+Mickiewicza&jednostka_miasto=&jednostka_wojewodztwo=&kierownik=&kierownik_plec=&kierownik_tytul=&status=&projekt=&kwotaprzyznanaod=8+375&kwotaprzyznanado=7+209+600&typkonkursu=&konkurs=&grupa=&panel=&slowokluczowe=&aparatura="

def get_driver():
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")  # działa w tle
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    return driver

def get_project_links(driver):
    driver.get(START_URL)
    time.sleep(3)  # poczekaj na załadowanie strony
    projekty = driver.find_elements(By.CSS_SELECTOR, "div.node.node-article h2 a")
    links = [p.get_attribute("href") for p in projekty]
    return links[:10]

def parse_project(driver, url):
    driver.get(url)
    time.sleep(2)

    data = {}

    # Tytuł
    try:
        data["tytul"] = driver.find_element(By.CSS_SELECTOR, ".important h2").text
    except:
        data["tytul"] = None

    # ID projektu
    try:
        data["id"] = driver.find_element(By.CSS_SELECTOR, ".important p.row2").text
    except:
        data["id"] = None

    # Słowa kluczowe
    data["slowa_kluczowe"] = [el.text for el in driver.find_elements(By.CSS_SELECTOR, "p.row2 span.frazy a")]

    # Deskryptory
    data["deskryptory"] = [el.text for el in driver.find_elements(By.CSS_SELECTOR, ".important ul li")]

    # Panel
    try:
        data["panel"] = driver.find_element(By.CSS_SELECTOR, "p.wciecie").text
    except:
        data["panel"] = None

    # Jednostka
    try:
        data["jednostka"] = driver.find_element(
            By.XPATH, "//p[strong[contains(text(), 'Jednostka realizująca')]]/following-sibling::p"
        ).text
    except:
        data["jednostka"] = None

    # Kierownik
    try:
        data["kierownik"] = driver.find_element(
            By.XPATH, "//p[strong[contains(text(), 'Kierownik projektu')]]/following-sibling::p"
        ).text
    except:
        data["kierownik"] = None

    # Szczegóły: kwota, daty, status
    for el in driver.find_elements(By.CSS_SELECTOR, ".strona p"):
        txt = el.text
        if "Przyznana kwota" in txt:
            data["kwota"] = txt
        elif "Rozpoczęcie projektu" in txt:
            data["start"] = txt
        elif "Zakończenie projektu" in txt:
            data["koniec"] = txt
        elif "Status projektu" in txt:
            data["status"] = txt

    # PDF
    try:
        data["opis_pdf"] = driver.find_element(By.CSS_SELECTOR, "a[href$='.pdf']").get_attribute("href")
    except:
        data["opis_pdf"] = None

    # Publikacje
    publikacje = []
    for pub in driver.find_elements(By.CSS_SELECTOR, "li.publikacje"):
        pub_data = {}
        try: pub_data["tytul"] = pub.find_element(By.CSS_SELECTOR, ".tytul strong").text
        except: pub_data["tytul"] = None
        try: pub_data["autorzy"] = pub.find_element(By.CSS_SELECTOR, ".autorzy em").text
        except: pub_data["autorzy"] = None
        try: pub_data["czasopismo"] = pub.find_element(By.CSS_SELECTOR, ".czasopismo em").text
        except: pub_data["czasopismo"] = None
        try: pub_data["doi"] = pub.find_element(By.CSS_SELECTOR, ".doi .prawa").text
        except: pub_data["doi"] = None
        publikacje.append(pub_data)

    data["publikacje"] = publikacje
    return data

def main():
    driver = get_driver()
    links = get_project_links(driver)
    print("Znalezione linki:", links)

    results = []
    for url in links:
        print("Scrapuję:", url)
        results.append(parse_project(driver, url))

    driver.quit()

    with open("projekty.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
