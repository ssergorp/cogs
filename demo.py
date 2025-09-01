#!/usr/bin/env python3
"""
COGS Demo Script
Demonstrates the COGS hybrid RAG system with example queries
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_rag_search import run_query

# Demo queries that showcase different capabilities
DEMO_QUERIES = [
    "Can I build a starbucks in the CS district?",
    "What building permits do I need for a renovation?", 
    "What are the setback requirements for residential buildings?",
    "Fire safety requirements for commercial buildings",
    "Parking requirements for retail establishments",
    "Height restrictions in residential zones",
    "What is required for a building permit?"
]

async def run_demo():
    """Run a demo of the COGS system with example queries"""
    
    print("🏗️ COGS - City Ordinance Guidance System Demo")
    print("=" * 60)
    print("This demo showcases COGS's ability to answer complex")
    print("questions about Dallas building codes using hybrid RAG.")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Use command line argument if provided
        query = " ".join(sys.argv[1:])
        await run_query(query)
    else:
        # Interactive demo mode
        print("\n🎯 Available demo queries:")
        for i, query in enumerate(DEMO_QUERIES, 1):
            print(f"{i}. {query}")
        
        print(f"{len(DEMO_QUERIES) + 1}. Enter your own question")
        print("0. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect a demo query (0-{len(DEMO_QUERIES) + 1}): ")
                
                if choice == "0":
                    print("👋 Thanks for trying COGS!")
                    break
                    
                elif choice == str(len(DEMO_QUERIES) + 1):
                    custom_query = input("Enter your building code question: ")
                    if custom_query.strip():
                        await run_query(custom_query.strip())
                    else:
                        print("❌ Please enter a valid question.")
                        continue
                        
                elif choice.isdigit() and 1 <= int(choice) <= len(DEMO_QUERIES):
                    selected_query = DEMO_QUERIES[int(choice) - 1]
                    await run_query(selected_query)
                else:
                    print("❌ Invalid selection. Please try again.")
                    continue
                    
                # Ask if user wants to continue
                continue_demo = input("\n🔄 Try another query? (y/n): ")
                if continue_demo.lower() not in ['y', 'yes']:
                    print("👋 Thanks for trying COGS!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Thanks for trying COGS!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                continue

def main():
    """Main entry point"""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Elasticsearch is running: curl http://localhost:9200")
        print("2. Check your MISTRAL_API_KEY in .env file")
        print("3. Verify building codes are indexed: curl http://localhost:9200/building_codes/_count")

if __name__ == "__main__":
    main()