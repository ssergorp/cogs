#!/usr/bin/env python3
"""
Fix embeddings that are stored as quoted JSON strings
"""

import sqlite3
import json
import argparse
from tqdm import tqdm

def fix_quoted_embeddings(db_path: str, batch_size: int = 1000, dry_run: bool = True):
    """Fix embeddings that are stored as quoted JSON strings"""

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get total count
    cur.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL AND embedding != ''")
    total_count = cur.fetchone()[0]

    print(f"🔍 Found {total_count} documents with embeddings to check")

    # Check a few samples first
    cur.execute("SELECT doc_id, embedding FROM documents WHERE embedding IS NOT NULL AND embedding != '' LIMIT 5")
    samples = cur.fetchall()

    print("\n📄 Sample embeddings:")
    valid_samples = 0
    invalid_samples = 0

    for doc_id, embedding_str in samples:
        print(f"\nDoc ID: {doc_id}")
        print(f"Raw value: {embedding_str[:100]}...")

        # Check if it's a quoted JSON string
        try:
            # First, try to parse it directly as JSON
            parsed_direct = json.loads(embedding_str)
            if isinstance(parsed_direct, list):
                print(f"  ✅ Already valid JSON array (dim: {len(parsed_direct)})")
                valid_samples += 1
                continue
        except:
            pass

        # Try to parse it as a quoted string containing JSON
        try:
            # Remove outer quotes if they exist
            if embedding_str.startswith('"') and embedding_str.endswith('"'):
                unquoted = embedding_str[1:-1]
                # Handle escaped quotes
                unquoted = unquoted.replace('\\"', '"')
                parsed_unquoted = json.loads(unquoted)
                if isinstance(parsed_unquoted, list):
                    print(f"  🔧 Needs unquoting - would become array (dim: {len(parsed_unquoted)})")
                    invalid_samples += 1
                    continue

            print(f"  ❌ Unknown format")

        except Exception as e:
            print(f"  ❌ Parse error: {e}")

    print(f"\n📊 Sample Analysis:")
    print(f"  ✅ Valid format: {valid_samples}")
    print(f"  🔧 Need unquoting: {invalid_samples}")

    if invalid_samples == 0:
        print("\n✅ All samples are already in valid format!")
        conn.close()
        return

    if dry_run:
        print(f"\n🧪 DRY RUN MODE - no changes will be made")
        print(f"   Run with --fix to actually update the database")
        conn.close()
        return

    # Actually fix the embeddings
    print(f"\n🔧 Fixing embeddings in batches of {batch_size}...")

    updated_count = 0
    error_count = 0

    # Process in batches
    for offset in tqdm(range(0, total_count, batch_size), desc="Processing batches"):
        cur.execute("""
            SELECT doc_id, embedding
            FROM documents
            WHERE embedding IS NOT NULL AND embedding != ''
            LIMIT ? OFFSET ?
        """, (batch_size, offset))

        batch = cur.fetchall()
        batch_updates = []

        for doc_id, embedding_str in batch:
            try:
                # Check if it's already valid
                parsed_direct = json.loads(embedding_str)
                if isinstance(parsed_direct, list):
                    continue  # Already valid
            except:
                pass

            try:
                # Try to unquote
                if embedding_str.startswith('"') and embedding_str.endswith('"'):
                    unquoted = embedding_str[1:-1]
                    unquoted = unquoted.replace('\\"', '"')
                    parsed_unquoted = json.loads(unquoted)

                    if isinstance(parsed_unquoted, list):
                        # This one needs fixing
                        fixed_embedding = json.dumps(parsed_unquoted)
                        batch_updates.append((fixed_embedding, doc_id))

            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Only print first 10 errors
                    print(f"Error processing {doc_id}: {e}")

        # Update this batch
        if batch_updates:
            cur.executemany(
                "UPDATE documents SET embedding = ? WHERE doc_id = ?",
                batch_updates
            )
            conn.commit()
            updated_count += len(batch_updates)

    conn.close()

    print(f"\n✅ Completed!")
    print(f"   Updated: {updated_count} embeddings")
    print(f"   Errors: {error_count} embeddings")

def verify_embeddings(db_path: str, sample_size: int = 10):
    """Verify that embeddings are now in correct format"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT doc_id, embedding
        FROM documents
        WHERE embedding IS NOT NULL AND embedding != ''
        LIMIT ?
    """, (sample_size,))

    samples = cur.fetchall()

    valid_count = 0
    dimensions = set()

    print(f"\n🔍 Verifying {len(samples)} sample embeddings:")

    for doc_id, embedding_str in samples:
        try:
            parsed = json.loads(embedding_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(x, (int, float)) for x in parsed[:5]):
                    valid_count += 1
                    dimensions.add(len(parsed))
                    print(f"  ✅ {doc_id}: valid array, dim {len(parsed)}")
                else:
                    print(f"  ❌ {doc_id}: array but not numeric")
            else:
                print(f"  ❌ {doc_id}: not an array")
        except Exception as e:
            print(f"  ❌ {doc_id}: parse error - {e}")

    success_rate = valid_count / len(samples) if samples else 0

    print(f"\n📊 Verification Results:")
    print(f"   Valid: {valid_count}/{len(samples)} ({success_rate:.1%})")
    print(f"   Dimensions found: {sorted(dimensions)}")

    conn.close()
    return success_rate > 0.8

def main():
    parser = argparse.ArgumentParser(description="Fix quoted embedding format")
    parser.add_argument("--db", default="data/cogs_memory.db", help="Database path")
    parser.add_argument("--fix", action="store_true", help="Actually fix the embeddings (default is dry run)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--verify-only", action="store_true", help="Only verify current embeddings")

    args = parser.parse_args()

    print(f"🔧 Embedding Format Fixer")
    print(f"Database: {args.db}")
    print("=" * 50)

    if args.verify_only:
        verify_embeddings(args.db)
        return

    fix_quoted_embeddings(args.db, args.batch_size, dry_run=not args.fix)

    if args.fix:
        print("\n🔍 Verifying fixes...")
        verify_embeddings(args.db)

if __name__ == "__main__":
    main()
