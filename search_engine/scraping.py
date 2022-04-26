import requests
from bs4 import BeautifulSoup
import random
import time
doc_id = 0


def get_text_from_link(url):
	response = requests.get(
		url=url,
	)
	soup = BeautifulSoup(response.content, features="html.parser")

	# kill all script and style elements
	for script in soup(["script", "style"]):
		script.extract()  # rip it out

	# get text
	text = soup.get_text()

	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)
	return text


def scrapeWikiArticle(url, max_links, i = 0):
	time.sleep(0.35)
	if i > max_links:
		return
	print(url)

	text = get_text_from_link(url)
	save_text_to_file(text, url)

	response = requests.get(
		url=url,
	)

	soup = BeautifulSoup(response.content, 'html.parser')

	title = soup.find(id="firstHeading")
	allLinks = soup.find(id="bodyContent").find_all("a")
	random.shuffle(allLinks)
	linkToScrape = 0

	for link in allLinks:
		if link['href']:
			if link['href'].find("/wiki/") == -1 or link['href'].find("en") == -1:
				continue

			linkToScrape = link
			break

	scrapeWikiArticle("https://en.wikipedia.org" + linkToScrape['href'], max_links, i+1)


def save_text_to_file(text, url):
	global doc_id
	file = open("documents/" + str(doc_id) + ".txt", 'w+')
	file.write(url + '\n')
	file.write(text)
	file.close()
	doc_id +=  1




if __name__ == '__main__':
	# with open("")
	url = 'https://en.wikipedia.org/wiki/Biology'
	get_text_from_link(url)
	scrapeWikiArticle(url, 12)
