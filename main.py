import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
import csv
import json
import flask_cors, flask
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings('ignore')

app = Flask(__name__)

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, key, value):
        new_node = Node(key, value)
        new_node.next = self.head
        self.head = new_node

    def search(self, key):
        current = self.head
        while current:
            if current.key == key:
                return current.value
            current = current.next
        return None

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        if self.table[index] is None:
            self.table[index] = LinkedList()
        self.table[index].insert(key, value)

    def search(self, key):
        index = self.hash_function(key)
        if self.table[index] is not None:
            return self.table[index].search(key)
        return None
    
def build_forward_index(data):
    forward_index = HashTable(size=100)
    for article_id, words in data["title"].items():
        forward_index.insert(article_id, words)
    return forward_index

def build_inverted_index(data):
    inverted_index = HashTable(size=100)
    for article_id, words in data["content"].items():
        for word in words:
            if inverted_index.search(word):
                inverted_index.search(word).append(article_id)
            else:
                inverted_index.insert(word, [article_id])
    return inverted_index


@app.route("/search_1", methods=["GET"])
def single_word_search(inverted_index, word):
    return inverted_index.search(word)


@app.route("/search", methods=["GET"])
def multi_word_search(inverted_index, query):
    result = set()
    words = query.split()
    if words:
        result = set(inverted_index.search(words[0])) if inverted_index.search(words[0]) else set()
        for word in words[1:]:
            current_result = inverted_index.search(word)
            if current_result:
                result.intersection_update(current_result)
    return list(result)


def rank_results(data, results):
    word_frequency = {}
    for article_id in results:
        for word in data["content"][article_id]:
            word_frequency[word] = word_frequency.get(word, 0) + 1
    return sorted(results, key=lambda x: sum(word_frequency[word] for word in data["content"][x]), reverse=True)

def display_results(data, results):
    for article_id in results:
        print(f"Article {article_id}: {data['content'][article_id]}")


@app.route("/add", methods=["GET"])
def add_content(data, new_article):
    article_id = str(len(data["index"]))
    data["index"][article_id] = len(data["index"])
    data["source"][article_id] = new_article[0]
    data["title"][article_id] = new_article[1]
    data["content"][article_id] = new_article[2]

    return data

if __name__ == "__main__":
    app.run(debug=True)