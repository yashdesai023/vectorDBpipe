
import argparse
import sys
import json
import os
from pathlib import Path
from vectorDBpipe.pipeline.text_pipeline import TextPipeline
from vectorDBpipe.config.config_manager import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="VectorDBPipe CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ingest_parser.add_argument("--source", default=None, help="Path to data source (file or directory)")
    
    # Query Command
    query_parser = subparsers.add_parser("query", help="Query the database")
    query_parser.add_argument("text", nargs="?", help="Query text (optional if interactive)") # Make text optional
    query_parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")
    query_parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode for faster queries")

    # Pipeline Run Command (Wrapper for ingest for now)
    pipeline_parser = subparsers.add_parser("pipeline", help="Manage pipelines")
    pipeline_parser.add_argument("action", choices=["run", "list", "stop"], help="Action")
    pipeline_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    if args.command == "ingest":
        try:
            print(f"Initializing pipeline with config: {args.config}")
            pipeline = TextPipeline(config_path=args.config)
            
            # Override data_path if source provided
            if args.source:
                # Ensure it's a Path object for the loader
                source_path = Path(args.source)
                pipeline.data_path = source_path
                pipeline.loader.data_path = source_path
                print(f"Overriding data source to: {args.source}")
                
            pipeline.process()
        except Exception as e:
            print(f"Error during ingestion: {e}")
            sys.exit(1)

    elif args.command == "query":
        try:
            pipeline = TextPipeline(config_path=args.config)

            if args.interactive:
                print("\n" + "="*40)
                print(" 🚀 Interactive Query Mode (Fast)")
                print(" Models loaded. Type 'exit' or 'q' to quit.")
                print("="*40 + "\n")
                
                while True:
                    try:
                        user_input = input("\n🔎 Query: ").strip()
                        if user_input.lower() in ['exit', 'quit', 'q']:
                            print("Bye!")
                            break
                        if not user_input:
                            continue
                            
                        results = pipeline.search(user_input, top_k=args.top_k)
                        
                        print(f"\nResults for: '{user_input}'")
                        if not results:
                            print("No results found.")
                        for i, res in enumerate(results):
                            score = res.get('score', 0)
                            meta = res.get('metadata', {})
                            text = meta.get('text', 'N/A')[:200].replace('\n', ' ')
                            source = meta.get('source', 'Unknown')
                            print(f"{i+1}. [{score:.4f}] {source}")
                            print(f"   \"{text}...\"")
                            
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                    except Exception as e:
                        print(f"Error: {e}")

            elif args.text:
                # One-shot query
                results = pipeline.search(args.text, top_k=args.top_k)
                
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print(f"\nSearch Results for: '{args.text}'\n")
                    if not results:
                        print("No results found.")
                    for i, res in enumerate(results):
                        score = res.get('score', 0)
                        meta = res.get('metadata', {})
                        text = meta.get('text', 'N/A')[:200].replace('\n', ' ')
                        source = meta.get('source', 'Unknown')
                        print(f"{i+1}. [{score:.4f}] {source}")
                        print(f"   \"{text}...\"\n")
            else:
                print("Error: query text is required unless using --interactive")
                sys.exit(1)

        except Exception as e:
            print(f"Error during query: {e}")
            sys.exit(1)

    elif args.command == "pipeline":
        if args.action == "run":
             # For now, pipeline run is just ingest
            print(f"Running pipeline (ingest) with config: {args.config}")
            pipeline = TextPipeline(config_path=args.config)
            pipeline.process()
        elif args.action == "list":
            print("Active Pipelines: [Default]")
        elif args.action == "stop":
            print("No active pipelines to stop.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
