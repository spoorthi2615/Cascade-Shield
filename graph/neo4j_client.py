"""
Helper functions for connecting to Neo4j and executing Cypher queries.
Uses credentials defined in NEO4J_AUTH environment variable.
"""
import os
from neo4j import GraphDatabase

# TODO: Initialize Neo4j driver using os.getenv("NEO4J_AUTH")
