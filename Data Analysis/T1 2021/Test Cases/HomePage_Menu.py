from selenium import webdriver
import time
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains

driver = webdriver.Chrome(executable_path="C:/Users/HP/Desktop/Project Tests/chromedriver.exe")
action = ActionChains(driver)


driver.get("http://localhost:5555/analysis")
driver.maximize_window()

assert "Energy Safe Victoria" in driver.title
driver.find_element_by_xpath("//img[@src='assets/img/logo.png']")

#driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#driver.execute_script("window.scrollTo(0, 100)")
time.sleep(3)


Home = driver.find_elements_by_xpath('//*[@href="#home"]')
for value in Home:
    print("Element exist -" + value.text)
about = driver.find_elements_by_xpath('//*[@href="#about"]')
for value in about:
    print("Element exist -" + value.text)
exploration = driver.find_elements_by_xpath('//*[@href="#exploration"]')
for value in exploration:
    print("Element exist -" + value.text)
visualization = driver.find_elements_by_xpath('//*[@href="#visualization"]')
for value in visualization:
    print("Element exist -" + value.text)
productPipeline = driver.find_elements_by_xpath('//*[@href="#productPipeline"]')
for value in productPipeline:
    print("Element exist -" + value.text)
Tools = driver.find_elements_by_xpath('//*[@href="#services"]')
for value in Tools:
    print("Element exist -" + value.text)
Team = driver.find_elements_by_xpath('//*[@href="#team-section"]')
for value in Team:
    print("Element exist -" + value.text)








expected_footer = " 309 - Energy Safe Victoria Head Office Level 5/4 Riverside Quay, Southbank VIC 3006"
footers = driver.find_elements_by_xpath("(//footer/div[@class='footer-center']/div/span)[1]")
for value in footers:
    print("Element exist -" + value.text)
if expected_footer in footers[0].text:
    print("text visible...")
else:
    pass

driver.quit