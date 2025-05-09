import os
import sys
from datetime import timedelta
import time
import chess
import matplotlib.pyplot as plt  # Importiamo matplotlib per il grafico

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mate_ia_one.engine import AlphaBetaEngine, MLEngine, PyTorchMLEngine, StockfishEngine
from mate_ia_one.modelNNUEpythorch import NNUEP
from mate_ia_one.modelNNUE import NNUE

TEST_REPORT_PATH = os.path.join(os.path.dirname(__file__), 'report.txt')


def generate_report(test_results):
    with open(TEST_REPORT_PATH, 'w') as report_file:
        report_file.write("ERET Test Suite Results\n")
        report_file.write("=" * 40 + "\n\n")

        for engine_result in test_results:
            report_file.write(f"Engine: {engine_result['engine_name']}\n")
            report_file.write(f"- Total Tests: {engine_result['total']}\n")
            report_file.write(f"- Correct: {engine_result['correct']} ({engine_result['success_rate']:.2f}%)\n")
            report_file.write(f"- Time: {str(timedelta(seconds=engine_result['elapsed']))}\n")

            if engine_result['failed_tests']:
                report_file.write("\nFailed Tests:\n")
                for fail in engine_result['failed_tests']:
                    report_file.write(f"ID: {fail['id']}\n")
                    report_file.write(f"Expected: {fail['expected']}\n")
                    report_file.write(f"Actual: {fail['actual']}\n\n")

            report_file.write("\n" + "=" * 40 + "\n\n")


# Load ERET test cases
eret_positions = []
with open('eret.epd', 'r') as f:
    for line in f:
        line = line.strip().replace(',', '')  # Pulisci le virgole
        if not line:
            continue
        try:
            board, ops = chess.Board.from_epd(line)

            # Cerca prima 'bm', poi 'am', infine gestisci casi mancanti
            moves_key = 'bm' if 'bm' in ops else 'am' if 'am' in ops else None

            if not moves_key:
                print(f"Warning: No 'bm'/'am' found in line: {line}")
                continue

            best_moves = ops[moves_key]
            bm_moves = [board.parse_uci(move.uci()) for move in best_moves]

            eret_positions.append({
                'fen': board.fen(),
                'bm': bm_moves,
                'id': ops['id'][0] if 'id' in ops else "NO_ID"
            })

        except Exception as e:
            print(f"Error parsing line: {line}\n{str(e)}")
            continue


def run_eret_test(engine_creator, computer_move_args=None):
    results = {
        'engine_name': engine_creator(chess.Board()).__class__.__name__,
        'total': len(eret_positions),
        'correct': 0,
        'failed_tests': [],
        'elapsed': 0,
        'success_rate': 0
    }

    start_time = time.time()
    computer_move_args = computer_move_args or {}
    print(f"Starting ERET Test for {results['engine_name']}")

    for position in eret_positions:
        print(f"Testing {position['fen']}")
        board = chess.Board(position['fen'])
        engine = engine_creator(board)
        try:
            engine_move = engine.computer_move(**computer_move_args)
        except Exception as e:
            print(f"Error in {position['id']}: {e}")
            engine_move = None

        if engine_move in position['bm']:
            print(f"Correct {position['id']}: {position['bm']}")
            results['correct'] += 1
        else:
            expected = ', '.join([move.uci() for move in position['bm']])
            actual = engine_move.uci() if engine_move else "None"
            results['failed_tests'].append({
                'id': position['id'],
                'expected': expected,
                'actual': actual
            })
            print(f"Failed {position['id']}: Expected {expected}, got {actual}")

    results['elapsed'] = time.time() - start_time
    results['success_rate'] = (results['correct'] / results['total']) * 100
    return results



if __name__ == "__main__":
    test_results = []
    result = {
        'engine_name': "AlphaBetaEngine",
        'total': 0,
        'correct': 3,
        'failed_tests': [],
        'elapsed': 0,
        'success_rate': 0
    }
    test_results.append(result)
    depth=10


    # Test AlphaBetaEngine
    print("Testing AlphaBetaEngine...")
    #test_results.append(run_eret_test(lambda board: AlphaBetaEngine(board)))

    # Test MLEngine
    print("\nTesting MLEngine...")
    model_nnue = NNUE.load('../mate_ia_one/weight/nnue_model.npz')
    test_results.append(run_eret_test(
        lambda board: MLEngine(board, model_nnue),
        {'depth': depth, 'beam_width': 2, 'temperature': 0}
    ))

    # Test PyTorchMLEngine
    print("\nTesting PyTorchMLEngine...")
    input_size = 769
    hidden_sizes = [769, 384, 192, 96]
    dropouts = 0.1
    model_nnuep = NNUEP.load('../weight/weight12/1500000_100_769.pth', input_size, hidden_sizes, dropouts)
    test_results.append(run_eret_test(
        lambda board: PyTorchMLEngine(board, model_nnuep),
        {'depth': depth, 'beam_width': 2, 'temperature': 0}
    ))

    # Test Stockfish
    print("\nTesting Stockfish...")
    stockfish_path = '/opt/homebrew/Cellar/stockfish/17/bin/stockfish'  # Modifica se necessario
    test_results.append(run_eret_test(
        lambda board: StockfishEngine(board, path=stockfish_path),
        {'depth': depth}
    ))

    generate_report(test_results)
    print(f"\nTest report generated at: {TEST_REPORT_PATH}")

    # --- Aggiungiamo la parte per il grafico a barre ---
    # Estraiamo i nomi dei motori e il numero delle risposte corrette
    engine_names = [result['engine_name'] for result in test_results]
    correct_counts = [result['correct'] for result in test_results]

    # Creiamo il grafico a barre
    plt.figure(figsize=(10, 6))
    bars = plt.bar(engine_names, correct_counts, color='skyblue')

    # Aggiungiamo il numero esatto sopra ogni barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{int(yval)}', ha='center', va='bottom')

    plt.xlabel('Engine')
    plt.ylabel('Numero di Risposte Corrette')
    plt.title('Risultati dei Test ERET per Engine')
    plt.ylim(0, max(correct_counts) + 5)
    plt.tight_layout()
    plt.savefig('./eret_graph_d'+str(depth)+'.png')
    plt.show()

