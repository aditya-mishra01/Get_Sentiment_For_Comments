from bs4 import BeautifulSoup
import requests


def get_comments(url):
    r=requests.get(url)
    commen_arr=[]
    soup = BeautifulSoup(r.content,'html5lib')
    
    if(url.split(".")[1]=="instagram"):
        comments= soup.find_all('div', class_="_a9zs")

        for comment in comments:
            commen_arr.append(comment.h5.text)
    elif(url.split(".")[1]=="facebook"):
        comments= soup.find_all('div', class_="xdj266r ")

        for comment in comments:
            commen_arr.append(comment.h5.text)
    else:
        raise ValueError("Url is invalid or not from instagram or facebook")
    return(commen_arr)

