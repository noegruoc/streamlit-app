"""Generate fake banking data with realistic transaction patterns."""

import string
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
NUM_CLIENTS = 10_000  # Reduced for meaningful relationships
NUM_TRANSACTIONS = 50_000
BATCH_SIZE = 50_000
DATA_DIR = Path(__file__).parent.parent / "data"

np.random.seed(42)

IBAN_CHARS = list(string.digits + string.ascii_uppercase)


def generate_ribs_fast(num_clients: int) -> np.ndarray:
    """Generate fake French IBANs using numpy."""
    print(f"Generating {num_clients:,} RIBs...")

    check_digits = np.random.randint(10, 100, size=num_clients)
    bank_codes = np.random.randint(10000, 100000, size=num_clients)
    branch_codes = np.random.randint(10000, 100000, size=num_clients)

    account_chars = np.random.choice(IBAN_CHARS, size=(num_clients, 11))
    account_numbers = np.array(["".join(row) for row in account_chars])

    ribs = np.array([
        f"FR{cd}{bc}{br}{acc}"
        for cd, bc, br, acc in zip(check_digits, bank_codes, branch_codes, account_numbers)
    ])

    print(f"  Done! Generated {len(ribs):,} RIBs")
    return ribs


def generate_clients(num_clients: int) -> pd.DataFrame:
    """Generate client data (id + rib only)."""
    ribs = generate_ribs_fast(num_clients)

    df = pd.DataFrame({
        "client_id": np.arange(1, num_clients + 1),
        "rib": ribs,
    })

    return df


def generate_transactions_realistic(
    client_ribs: np.ndarray,
    num_transactions: int,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Generate transactions with realistic patterns (repeat relationships)."""
    num_clients = len(client_ribs)
    print(f"Generating {num_transactions:,} transactions with realistic patterns...")

    # Create client activity weights (power law: some clients very active)
    # This makes ~20% of clients responsible for ~80% of transactions
    activity_weights = np.random.pareto(a=1.5, size=num_clients) + 1
    activity_weights = activity_weights / activity_weights.sum()

    # Pre-generate some "preferred partners" for each client
    # Each client has 3-20 preferred partners they transact with more often
    num_partners = np.random.randint(3, 20, size=num_clients)
    preferred_partners = {}
    all_indices = np.arange(num_clients)
    for i in range(num_clients):
        # Choose partners weighted by their activity (excluding self)
        mask = all_indices != i
        other_indices = all_indices[mask]
        other_weights = activity_weights[mask]
        other_weights = other_weights / other_weights.sum()
        partners = np.random.choice(
            other_indices,
            size=min(num_partners[i], len(other_indices)),
            replace=False,
            p=other_weights
        )
        preferred_partners[i] = partners

    # Generate transactions
    senders = []
    receivers = []

    for _ in range(num_transactions):
        # Choose sender weighted by activity
        sender_idx = np.random.choice(num_clients, p=activity_weights)

        # 70% chance to use preferred partner, 30% random
        if np.random.random() < 0.7 and len(preferred_partners[sender_idx]) > 0:
            receiver_idx = np.random.choice(preferred_partners[sender_idx])
        else:
            receiver_idx = np.random.choice(num_clients)
            while receiver_idx == sender_idx:
                receiver_idx = np.random.choice(num_clients)

        senders.append(sender_idx)
        receivers.append(receiver_idx)

    senders = np.array(senders)
    receivers = np.array(receivers)

    # Generate amounts (lognormal distribution)
    amounts = np.round(np.random.lognormal(mean=4.5, sigma=1.5, size=num_transactions), 2)
    amounts = np.clip(amounts, 1, 100_000)

    # Generate dates
    date_range_days = (end_date - start_date).days
    random_days = np.random.randint(0, date_range_days, size=num_transactions)
    random_seconds = np.random.randint(0, 86400, size=num_transactions)
    transaction_dates = [
        start_date + timedelta(days=int(d), seconds=int(s))
        for d, s in zip(random_days, random_seconds)
    ]

    # Transaction types
    tx_types = np.random.choice(
        ["virement", "prelevement", "paiement_carte", "retrait"],
        size=num_transactions,
        p=[0.4, 0.25, 0.25, 0.1],
    )

    df = pd.DataFrame({
        "transaction_id": np.arange(1, num_transactions + 1),
        "emetteur_rib": client_ribs[senders],
        "destinataire_rib": client_ribs[receivers],
        "montant": amounts,
        "transaction_date": transaction_dates,
        "type_transaction": tx_types,
    })

    # Print stats
    edge_counts = df.groupby(["emetteur_rib", "destinataire_rib"]).size()
    print(f"  Unique relationships: {len(edge_counts):,}")
    print(f"  Max transactions between same pair: {edge_counts.max()}")
    print(f"  Pairs with >1 transaction: {(edge_counts > 1).sum():,}")
    print(f"  Pairs with >5 transactions: {(edge_counts > 5).sum():,}")

    return df


def main() -> None:
    """Main function to generate all data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAKE BANKING DATA GENERATOR (Realistic Patterns)")
    print("=" * 60)
    print(f"Clients: {NUM_CLIENTS:,}")
    print(f"Transactions: {NUM_TRANSACTIONS:,}")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)

    # Generate clients
    clients_df = generate_clients(NUM_CLIENTS)
    clients_path = DATA_DIR / "clients.parquet"
    clients_df.to_parquet(clients_path, index=False)
    print(f"Saved clients to {clients_path}")

    client_ribs = clients_df["rib"].values
    del clients_df

    # Generate transactions
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    transactions_dir = DATA_DIR / "transactions"
    transactions_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files
    for f in transactions_dir.glob("*.parquet"):
        f.unlink()

    tx_df = generate_transactions_realistic(
        client_ribs, NUM_TRANSACTIONS, start_date, end_date
    )

    output_path = transactions_dir / "transactions_batch_000.parquet"
    tx_df.to_parquet(output_path, index=False)
    print(f"Saved transactions to {output_path}")

    print("=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
