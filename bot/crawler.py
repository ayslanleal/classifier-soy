import os
import json
import requests
from bs4 import BeautifulSoup

GOOGLE_SEARCH = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
SAVE_FOLDER = ['Ácaros-Verde', 'Bicudo-da-soja']


def main():
    for i in SAVE_FOLDER:
        if not os.path.exists(i):
            os.mkdir(i)
        download_image(i)


def download_image(search):
    # Obeter link
    searchURL = GOOGLE_SEARCH + 'q=' + search

    response = requests.get(searchURL)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    result = soup.findAll('img', {'class': 't0fcAb'}, limit=20)
    image_links = []
    for i in result:
        image_links.append(i['src'])
    for x, y in enumerate(image_links):
        response = requests.get(y)
        image_name = search + '/' + 'image' + str(x + 1) + '.jpg'
        with open(image_name, 'wb') as file:
            file.write(response.content)


if __name__ == '__main__':
    response = requests.get(GOOGLE_SEARCH+'q='+'Ácaros-Verde')
    html = response.text
    soup = BeautifulSoup(html,'html.parser')
    result = soup.findAll('img', {'class': 'rg_i Q4LuWd'}, limit=20)
    print(result)