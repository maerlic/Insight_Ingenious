#!/usr/bin/env python3
"""
Script to populate Azure AI Search with dummy data for testing the knowledge-base-agent.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Search configuration
service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
key = os.getenv("AZURE_AI_SEARCH_KEY")
index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "ingenious-kb-index")

if not service_endpoint or not key:
    print("Error: AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_KEY must be set in .env")
    sys.exit(1)

# Create clients
credential = AzureKeyCredential(key)
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
search_client = SearchClient(
    endpoint=service_endpoint, index_name=index_name, credential=credential
)


# Define the index schema
def create_index() -> None:
    """Create the search index if it doesn't exist."""
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="title", type="Edm.String", filterable=True),
        SearchableField(name="content", type="Edm.String"),
        SearchableField(
            name="category", type="Edm.String", filterable=True, facetable=True
        ),
        SimpleField(name="metadata", type="Edm.String"),
    ]

    # Configure semantic search
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    # Create the search index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    try:
        index_client.create_or_update_index(index)
        print(f"Index '{index_name}' created or updated successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")
        sys.exit(1)


# Sample documents
documents = [
    {
        "id": str(uuid.uuid4()),
        "title": "Azure Configuration Guide",
        "content": "This guide covers the essential steps for configuring Azure services including Azure AI Search, Azure SQL Database, and Azure OpenAI. First, ensure you have an active Azure subscription. Then, create resource groups to organize your services. For Azure AI Search, you'll need to create a search service, define indexes, and configure semantic search capabilities. Remember to secure your API keys and use environment variables for configuration.",
        "category": "Azure configuration and setup",
        "metadata": "Technical documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Workplace Safety Guidelines",
        "content": "Workplace safety is paramount for all employees. Always wear appropriate personal protective equipment (PPE) in designated areas. Report any hazards immediately to your supervisor. Know the location of emergency exits, fire extinguishers, and first aid kits. Participate in regular safety training sessions. Follow proper lifting techniques to prevent injuries. Keep work areas clean and organized to prevent trips and falls.",
        "category": "Workplace safety guidelines",
        "metadata": "Safety documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Nutrition and Health Information",
        "content": "Maintaining good nutrition is essential for overall health and wellbeing. Aim for a balanced diet that includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Stay hydrated by drinking at least 8 glasses of water daily. Limit processed foods, excessive sugar, and saturated fats. Regular meal timing helps maintain stable blood sugar levels. Consider consulting with a nutritionist for personalized dietary advice.",
        "category": "Health information and nutrition",
        "metadata": "Health documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Emergency Procedures Manual",
        "content": "In case of emergency, remain calm and follow established procedures. For fire emergencies, activate the nearest fire alarm and evacuate using designated routes. For medical emergencies, call emergency services immediately and provide first aid if trained. During severe weather, move to designated safe areas. Keep emergency contact numbers readily available. Participate in regular emergency drills to stay prepared.",
        "category": "Emergency procedures",
        "metadata": "Emergency documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Mental Health and Wellbeing Resources",
        "content": "Mental health is as important as physical health. Recognize signs of stress, anxiety, and burnout. Take regular breaks during work to prevent fatigue. Practice mindfulness and relaxation techniques. Maintain work-life balance by setting boundaries. Seek support from employee assistance programs when needed. Remember that asking for help is a sign of strength, not weakness. Regular exercise and adequate sleep contribute to mental wellbeing.",
        "category": "Mental health and wellbeing",
        "metadata": "Wellbeing documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "First Aid Basics",
        "content": "Basic first aid knowledge can save lives. Learn to recognize signs of medical emergencies. For bleeding, apply direct pressure with a clean cloth. For burns, cool the area with running water. Know how to perform CPR and use an AED. Keep first aid supplies accessible and check expiration dates regularly. Document all first aid incidents. Remember to protect yourself with gloves when providing aid.",
        "category": "First aid basics",
        "metadata": "Medical documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Azure OpenAI Integration Guide",
        "content": "Integrating Azure OpenAI into your applications requires proper setup and configuration. Start by creating an Azure OpenAI resource in your subscription. Request access to the models you need. Configure API keys and endpoints in your application. Use the latest SDK versions for best compatibility. Implement proper error handling and retry logic. Monitor usage to control costs. Consider implementing content filtering for production applications.",
        "category": "Azure configuration and setup",
        "metadata": "Technical documentation",
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Ergonomics in the Workplace",
        "content": "Proper ergonomics prevents workplace injuries and improves productivity. Adjust your chair height so feet are flat on the floor. Position your monitor at eye level to prevent neck strain. Keep frequently used items within easy reach. Take micro-breaks every hour to stretch. Use ergonomic keyboards and mice to prevent repetitive strain injuries. Report any discomfort early to prevent chronic conditions.",
        "category": "Workplace safety guidelines",
        "metadata": "Safety documentation",
    },
]


def upload_documents() -> None:
    """Upload documents to Azure Search."""
    try:
        result = search_client.upload_documents(documents=documents)
        print(f"Uploaded {len(documents)} documents successfully.")

        # Check for any failed uploads
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"Failed to upload {len(failed)} documents:")
            for f in failed:
                print(f"  - Document {f.key}: {f.error_message}")
    except Exception as e:
        print(f"Error uploading documents: {e}")
        sys.exit(1)


def verify_upload() -> None:
    """Verify documents were uploaded by performing a search."""
    try:
        results = search_client.search(
            search_text="*", select=["title", "category"], include_total_count=True
        )

        print(f"\nVerification: Found {results.get_count()} documents in the index:")
        for result in results:
            print(f"  - {result['title']} (Category: {result['category']})")
    except Exception as e:
        print(f"Error verifying upload: {e}")


if __name__ == "__main__":
    print(f"Connecting to Azure Search at: {service_endpoint}")
    print(f"Using index: {index_name}\n")

    # Create index
    create_index()

    # Upload documents
    print("\nUploading dummy documents...")
    upload_documents()

    # Verify upload
    print("\nVerifying upload...")
    verify_upload()

    print("\nDone! The knowledge-base-agent can now be tested with Azure AI Search.")
