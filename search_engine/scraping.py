from base64 import encode
import requests
from bs4 import BeautifulSoup
import random
import time
doc_id = 82

#Category
#Help
#Template
#png

def get_text_from_link(response):
	try:
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
	except:
		time.sleep(5)
		return get_text_from_link(url)


def scrapeWikiArticle(url, max_links, i=0):
	time.sleep(0.1)
	if i > max_links:
		return

	try:
		response = requests.get(
			url=url,
		)
	except:
		return
	print(url)

	text = get_text_from_link(response)
	

	save_text_to_file(text, url)

	soup = BeautifulSoup(response.content, 'html.parser')

	# title = soup.find(id="firstHeading")
	allLinks = soup.find(id="bodyContent").find_all("a")
	random.shuffle(allLinks)

	for link in allLinks:
		try:
			if link['href']:
				if link['href'].find("/wiki/") == -1 or\
				   link['href'].find("en") == -1 or\
				   link['href'].find("Category") != -1 or\
				   link['href'].find("File") != -1 or\
				   link['href'].find("png") != -1 or\
				   link['href'].find("jpg") != -1 or\
				   link['href'].find("Template") != -1 or\
				   link['href'].find("Help") != -1:
					continue

				linkToScrape = link
				scrapeWikiArticle("https://en.wikipedia.org" + linkToScrape['href'], max_links, i+1)
				break
		except:
			continue

	# scrapeWikiArticle("https://en.wikipedia.org" + linkToScrape['href'], max_links, i+1)


def save_text_to_file(text, url):
	global doc_id
	filename = url[url.find("wiki/") + len("wiki/"):]
	file = open("documents/" + filename + ".txt", 'w+', encoding='utf8')
	file.write(url + '\n')
	file.write(text)
	file.close()
	doc_id +=  1


if __name__ == '__main__':
	starting_urls = [
		'https://en.wikipedia.org/wiki/Mathematics',
		'https://en.wikipedia.org/wiki/Biology',
		'https://en.wikipedia.org/wiki/Physics',
		'https://en.wikipedia.org/wiki/Car',
		'https://en.wikipedia.org/wiki/History',
		'https://en.wikipedia.org/wiki/Esports',
		'https://en.wikipedia.org/wiki/Music',
		'https://en.wikipedia.org/wiki/Sport',
		'https://en.wikipedia.org/wiki/Dog',
		'https://en.wikipedia.org/wiki/Geography',
		'https://en.wikipedia.org/wiki/English_Wikipedia'
	]

	for i in range(10):
		random.shuffle(starting_urls)
		for url in starting_urls[::-1]:
			scrapeWikiArticle(url, 2000)
