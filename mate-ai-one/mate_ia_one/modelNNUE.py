import os
import re
from collections import defaultdict
from functools import lru_cache
import chess
import csv
import numpy as np
from chess.pgn import read_game
from io import StringIO
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from stockfish import Stockfish

# Configurazione filtro ELO
MIN_ELO = 1800
MAX_ELO_DIFF = 100

# Configurazione campionamento
SAMPLE_EVERY_N_MOVES = 5
TARGET_SAMPLES = 5000


#########################################
# Modello NNUE con dropout (implementato in numpy)
#########################################
class NNUE:
    def __init__(self, input_size=768, hidden_size=768, dropout_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.dropout_rate = dropout_rate  # Parametro per il dropout

        std_dev_input_to_hidden = np.sqrt(2 / self.input_size)
        std_dev_hidden_to_output = np.sqrt(2 / self.hidden_size)

        self.input_to_hidden_weights = np.random.randn(self.input_size, self.hidden_size) * std_dev_input_to_hidden
        self.hidden_to_output_weights = np.random.randn(self.hidden_size, self.output_size) * std_dev_hidden_to_output

        self.hidden_bias = np.zeros(self.hidden_size)
        self.output_bias = np.zeros(self.output_size)

    def evaluate(self, input_vector, training=False):
        # Forward pass: calcolo del layer nascosto
        hidden_pre = np.dot(input_vector, self.input_to_hidden_weights) + self.hidden_bias
        hidden = np.maximum(0, hidden_pre)  # Attivazione ReLU
        if training:
            # Applica dropout: genera una maschera casuale e normalizza
            mask = (np.random.rand(*hidden.shape) > self.dropout_rate).astype(np.float32)
            hidden = hidden * mask / (1 - self.dropout_rate)
        output = np.dot(hidden, self.hidden_to_output_weights) + self.output_bias
        return output

    def save_weights(self, filename):
        np.savez(filename,
                 input_to_hidden_weights=self.input_to_hidden_weights,
                 hidden_to_output_weights=self.hidden_to_output_weights,
                 hidden_bias=self.hidden_bias,
                 output_bias=self.output_bias)

    @staticmethod
    def load(filename):
        data = np.load(filename)
        nnue = NNUE()
        nnue.input_to_hidden_weights = data['input_to_hidden_weights']
        nnue.hidden_to_output_weights = data['hidden_to_output_weights']
        nnue.hidden_bias = data['hidden_bias']
        nnue.output_bias = data['output_bias']
        return nnue


#########################################
# Trainer con weight decay ed early stopping
#########################################
class NNUE_Trainer:
    def __init__(self, nnue):
        self.nnue = nnue

    def train(self, X, y, learning_rate=0.001, epochs=100, weight_decay=1e-4, patience=5):
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            # Per ogni esempio di training, eseguiamo la forward e backprop manuale
            for i in range(len(X)):
                input_vector = X[i].flatten().astype(np.float32)
                target = y[i].astype(np.float32)

                # Forward pass in modalità training (con dropout)
                output = self.nnue.evaluate(input_vector, training=True)
                error = target - output[0]
                total_loss += error ** 2

                output_gradient = error  # scala (gradiente dell'output)
                # Calcolo dei gradienti per hidden_to_output e output_bias
                hidden_pre = np.dot(input_vector, self.nnue.input_to_hidden_weights) + self.nnue.hidden_bias
                hidden = np.maximum(0, hidden_pre)
                grad_hidden_to_output = hidden[:, None] * output_gradient  # forma (hidden_size, 1)
                grad_output_bias = np.array([output_gradient])

                # Gradiente per il layer nascosto
                d_hidden = (hidden_pre > 0).astype(np.float32)
                grad_hidden = self.nnue.hidden_to_output_weights[:, 0] * output_gradient * d_hidden
                grad_input_to_hidden = np.outer(input_vector, grad_hidden)
                grad_hidden_bias = grad_hidden

                # Aggiornamento dei pesi con weight decay (L2 regolarizzazione)
                self.nnue.input_to_hidden_weights += learning_rate * (
                            grad_input_to_hidden - weight_decay * self.nnue.input_to_hidden_weights)
                self.nnue.hidden_bias += learning_rate * (grad_hidden_bias - weight_decay * self.nnue.hidden_bias)
                self.nnue.hidden_to_output_weights += learning_rate * (
                            grad_hidden_to_output - weight_decay * self.nnue.hidden_to_output_weights)
                self.nnue.output_bias += learning_rate * (grad_output_bias - weight_decay * self.nnue.output_bias)

            avg_loss = total_loss / len(X)
            predictions = np.array([self.nnue.evaluate(x)[0] for x in X], dtype=np.float32)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100  # Avoid division by zero
            print(
                f"Epoch {epoch + 1}/{epochs} - MSE: {mse:.4f} - RMSE: {rmse:.4f} - MAE: {mae:.4f} - R²: {r2:.4f} - MAPE: {mape:.2f}%")

            # Early stopping basato sul MSE (si potrebbe usare un validation set)
            current_loss = mse
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping attivato dopo {epoch + 1} epoche!")
                    break

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def evaluate(self, X, y, phase="test"):
        predictions = np.array([self.nnue.evaluate(x)[0] for x in X], dtype=np.float32)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100
        print(
            f"{phase.capitalize()} Metrics - MSE: {mse:.4f} - RMSE: {rmse:.4f} - MAE: {mae:.4f} - R²: {r2:.4f} - MAPE: {mape:.2f}%")
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


#########################################
# Resto delle funzioni (conversione FEN, parsing, campionamento)
#########################################
def fen_to_input_vector(fen):
    board = chess.Board(fen)
    input_vector = np.zeros(768)
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            piece_char = piece.symbol()
            feature_index = piece_to_index[piece_char] * 64 + square
            input_vector[feature_index] = 1
    return input_vector


def parse_an_string(an):
    try:
        cleaned_an = re.sub(r'\{.*?\}', '', an)
        cleaned_an = re.sub(r'\$\d+', '', cleaned_an)
        cleaned_an = cleaned_an.strip()
        pgn_stream = StringIO(cleaned_an)
        game = read_game(pgn_stream)
        if not game:
            return []
        main_line = []
        node = game
        while node.variations:
            next_node = node.variation(0)
            main_line.append(next_node.move.uci())
            node = next_node
        return main_line
    except Exception as e:
        print(f"Errore nel parsing della partita: {str(e)}")
        return []


def is_critical_position(board):
    if board.is_check():
        return True
    if board.is_capture(board.peek()):
        return True
    if board.peek().promotion:
        return True
    attacked_squares = board.attacks(board.king(not board.turn))
    if len(attacked_squares) > 1:
        return True
    material_diff = abs(chess.Material(board).white_material - chess.Material(board).black_material)
    if material_diff > 150:
        return True
    return False


def sample_positions(positions):
    from math import ceil
    phase_buckets = defaultdict(list)
    for fen, eval_score in positions:
        board = chess.Board(fen)
        piece_count = len(board.piece_map())
        if piece_count > 30:
            phase = 'opening'
        elif piece_count > 12:
            phase = 'middlegame'
        else:
            phase = 'endgame'
        phase_buckets[phase].append((fen, eval_score))
    total = max(1, len(positions))
    sampled = []
    for phase, items in phase_buckets.items():
        if not items:
            continue
        target = ceil(TARGET_SAMPLES * (len(items) / total))
        target = min(target, len(items))
        indices = np.arange(len(items))
        try:
            selected_indices = np.random.choice(indices, size=target, replace=(target > len(items)))
            sampled += [items[i] for i in selected_indices]
        except Exception as e:
            print(f"Errore campionamento {phase}: {str(e)}")
            continue
    try:
        return np.random.permutation(sampled)[:TARGET_SAMPLES].tolist()
    except:
        return sampled[:TARGET_SAMPLES]


def collect_data_from_csv(filename, stockfish_path):
    data = []
    skipped = {'invalid_rows': 0, 'parsing_errors': 0, 'invalid_moves': 0, 'sampled_positions': 0}
    try:
        stockfish = Stockfish(path=stockfish_path)
        stockfish.set_depth(15)
    except FileNotFoundError:
        print(f"Errore: Stockfish non trovato al percorso '{stockfish_path}'. Verifica il percorso.")
        return []
    except Exception as e:
        print(f"Errore inizializzazione Stockfish: {str(e)}")
        return []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        total_games = 0
        valid_games = 0

        for row in reader:
            total_games += 1
            if not validate_row(row):
                skipped['invalid_rows'] += 1
                continue
            try:
                an = row['AN'].strip()
            except (KeyError, ValueError) as e:
                print(f"Errore estrazione dati riga {total_games}: {str(e)}")
                skipped['invalid_rows'] += 1
                continue
            try:
                moves = parse_an_string(an)
                if not moves:
                    skipped['parsing_errors'] += 1
                    continue
            except Exception as e:
                print(f"Errore parsing AN riga {total_games}: {str(e)}")
                skipped['parsing_errors'] += 1
                continue

            board = chess.Board()
            positions = []
            critical_count = 0

            try:
                for i, move_uci in enumerate(moves):
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            raise ValueError(f"Mossa illegale: {move_uci}")
                        board.push(move)
                        if (i < 10 or i % SAMPLE_EVERY_N_MOVES == 0 or is_critical_position(
                                board) or board.is_game_over()):
                            if board.is_checkmate():
                                evaluation = 1.0 if board.turn else -1.0
                            elif board.is_stalemate() or board.is_insufficient_material():
                                evaluation = 0.0
                            else:
                                stockfish.set_fen_position(board.fen())
                                eval_dict = stockfish.get_evaluation()
                                if eval_dict['type'] == 'cp':
                                    eval_value = eval_dict['value'] / 1000.0
                                    evaluation = max(min(eval_value, 1.0), -1.0)
                                elif eval_dict['type'] == 'mate':
                                    evaluation = 1.0 if eval_dict['value'] > 0 else -1.0
                                else:
                                    evaluation = 0.0
                            positions.append((board.fen(), evaluation))
                            skipped['sampled_positions'] += 1
                            if is_critical_position(board):
                                critical_count += 1
                    except Exception as e:
                        print(f"Errore mossa {i + 1} ({move_uci}): {str(e)}")
                        skipped['invalid_moves'] += 1
                        break
                data.extend(positions)
                valid_games += 1
                if valid_games % 10 == 0:
                    print(f"Progresso: {valid_games} partite valide | Total: {total_games} | Posizioni: {len(data)}")
            except Exception as e:
                print(f"Errore processamento partita {total_games}: {str(e)}")
                skipped['invalid_moves'] += 1

            if os.getenv('DEBUG') and valid_games >= 100:
                break

    print("Bilanciamento posizioni...")
    balanced_data = sample_positions(data)
    print("Salvataggio risultati...")
    with open('datapickle/chess_data.pickle', 'wb') as f:
        pickle.dump(balanced_data, f)

    print("\n=== Report finale ===")
    print(f"Partite totali: {total_games}")
    print(f"Partite valide: {valid_games} ({valid_games / total_games:.1%})")
    print(f"Posizioni campionate: {len(balanced_data)}")
    print("\nMotivi esclusione:")
    print(f"- Righe non valide: {skipped['invalid_rows']}")
    print(f"- Errori parsing: {skipped['parsing_errors']}")
    print(f"- Mosse non valide: {skipped['invalid_moves']}")
    print(f"- Posizioni campionate: {skipped['sampled_positions']}")
    return balanced_data


def validate_row(row):
    required_fields = {'AN', 'WhiteElo', 'BlackElo'}
    missing = [field for field in required_fields if field not in row]
    if missing:
        print(f"Campi mancanti {missing} nella riga")
        return False
    try:
        white_elo = int(row['WhiteElo'])
        black_elo = int(row['BlackElo'])
    except ValueError:
        print(f"Formato ELO non valido: Bianco='{row['WhiteElo']}', Nero='{row['BlackElo']}'")
        return False
    if white_elo < MIN_ELO or black_elo < MIN_ELO:
        print(f"ELO troppo basso: Bianco={white_elo}, Nero={black_elo}")
        return False
    if abs(white_elo - black_elo) > MAX_ELO_DIFF:
        print(f"Differenza ELO eccessiva: {abs(white_elo - black_elo)}")
        return False
    return True


def train_model_from_data():
    with open('./datapickle/chess_data_3000.pickle', 'rb') as f:
        data = pickle.load(f)
    clean_data = []
    for fen, eval_val in data:
        try:
            clean_eval = float(eval_val)
            clean_data.append((fen, clean_eval))
        except (TypeError, ValueError) as e:
            print(f"Valore non valido scartato: {eval_val} - Errore: {str(e)}")
    if not clean_data:
        raise ValueError("Nessun dato valido disponibile per il training")

    X = [fen_to_input_vector(fen) for fen, _ in clean_data]
    y = [eval_val for _, eval_val in clean_data]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verifica se il file dei pesi esiste già.
    if os.path.exists('weight/nnue_model.npz'):
        print("Carico pesi esistenti da 'nnue_model.npz'...")
        nnue = NNUE.load('weight/nnue_model.npz')
    else:
        print("Nessun file di pesi trovato. Creo un nuovo modello NNUE.")
        nnue = NNUE()
    trainer = NNUE_Trainer(nnue)

    print("\nAvvio training...")
    train_metrics = trainer.train(X_train, y_train, learning_rate=0.001, epochs=50)

    print("\nValutazione sul set di training...")
    train_eval_metrics = trainer.evaluate(X_train, y_train, phase="train")
    print("\nValutazione sul set di test...")
    test_metrics = trainer.evaluate(X_test, y_test, phase="test")

    # Salva sempre i pesi al termine del training.
    nnue.save_weights('nnue_model.npz')

    # Generazione report (opzionale)
    with open('../report/training/training_report.txt', 'w') as f:
        f.write("=== Report di Addestramento e Valutazione ===\n\n")
        f.write("Dataset:\n")
        f.write(f" - Dimensioni totali: {len(X)}\n")
        f.write(f" - Training set: {len(X_train)} ({len(X_train) / len(X):.1%})\n")
        f.write(f" - Test set: {len(X_test)} ({len(X_test) / len(X):.1%})\n\n")
        f.write("Metriche di Training (ultima epoca):\n")
        f.write(f" - MSE: {train_metrics['mse']:.4f}\n")
        f.write(f" - RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f" - MAE: {train_metrics['mae']:.4f}\n")
        f.write(f" - R²: {train_metrics['r2']:.4f}\n")
        f.write(f" - MAPE: {train_metrics['mape']:.2f}%\n\n")
        f.write("Metriche di Valutazione (Training Set):\n")
        f.write(f" - MSE: {train_eval_metrics['mse']:.4f}\n")
        f.write(f" - RMSE: {train_eval_metrics['rmse']:.4f}\n")
        f.write(f" - MAE: {train_eval_metrics['mae']:.4f}\n")
        f.write(f" - R²: {train_eval_metrics['r2']:.4f}\n")
        f.write(f" - MAPE: {train_eval_metrics['mape']:.2f}%\n\n")
        f.write("Metriche di Valutazione (Test Set):\n")
        f.write(f" - MSE: {test_metrics['mse']:.4f}\n")
        f.write(f" - RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f" - MAE: {test_metrics['mae']:.4f}\n")
        f.write(f" - R²: {test_metrics['r2']:.4f}\n")
        f.write(f" - MAPE: {test_metrics['mape']:.2f}%\n")


def train_model_with_shuffling_and_cv(random_seed=42, k_folds=5, epochs=10):
    """
    Addestra il modello con random shuffling e k-fold cross-validation,
    quindi genera un report con le metriche medie sui fold.

    Parametri:
    - random_seed: Seme per il random shuffling (default: 42).
    - k_folds: Numero di fold per la cross-validation (default: 5).
    - epochs: Numero di epoche per l'addestramento (default: 100).
    """
    # Carica i dati
    with open('datapickle/chess_data.pickle', 'rb') as f:
        data = pickle.load(f)

    # Pulizia dei dati
    clean_data = []
    for fen, eval_val in data:
        try:
            clean_eval = float(eval_val)
            clean_data.append((fen, clean_eval))
        except (TypeError, ValueError) as e:
            print(f"Valore non valido scartato: {eval_val} - Errore: {str(e)}")
    if not clean_data:
        raise ValueError("Nessun dato valido disponibile per il training")

    # Converti i dati in X (input) e y (target)
    X = [fen_to_input_vector(fen) for fen, _ in clean_data]
    y = [eval_val for _, eval_val in clean_data]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Random shuffling con seme configurabile
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Configurazione della k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []

    # Esecuzione della cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Crea un nuovo modello per ogni fold
        model = NNUE()
        trainer = NNUE_Trainer(model)

        # Addestramento
        print("Avvio training...")
        trainer.train(X_train, y_train, epochs=epochs)

        # Valutazione sul set di test del fold
        print("Valutazione sul set di test...")
        test_metrics = trainer.evaluate(X_test, y_test, phase="test")
        fold_metrics.append(test_metrics)

    # Calcolo delle metriche medie sui fold
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics]),
        'mape': np.mean([m['mape'] for m in fold_metrics])
    }

    print("\n=== Metriche Medie su Tutti i Fold ===")
    print(f"MSE: {avg_metrics['mse']:.4f}")
    print(f"RMSE: {avg_metrics['rmse']:.4f}")
    print(f"MAE: {avg_metrics['mae']:.4f}")
    print(f"R²: {avg_metrics['r2']:.4f}")
    print(f"MAPE: {avg_metrics['mape']:.2f}%")

    # Generazione del report in formato txt
    report_path = f'../report/training/training_report_nnue_seed_{random_seed}.txt'
    with open(report_path, 'w') as f:
        f.write("=== Report di Cross-Validation ===\n\n")
        f.write("Dataset:\n")
        f.write(f" - Dimensioni totali: {len(X)}\n\n")
        f.write(f"Metriche Medie della Cross-Validation (su {k_folds} fold):\n")
        f.write(f" - MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f" - RMSE: {avg_metrics['rmse']:.4f}\n")
        f.write(f" - MAE: {avg_metrics['mae']:.4f}\n")
        f.write(f" - R²: {avg_metrics['r2']:.4f}\n")
        f.write(f" - MAPE: {avg_metrics['mape']:.2f}%\n")

    print(f"Report salvato in '{report_path}'")

if __name__ == "__main__":
    csv_filename = './dataset/chess_games.csv'  # Percorso fisso per il file CSV
    stockfish_path = '/opt/homebrew/Cellar/stockfish/17/bin/stockfish'  # Percorso fisso per Stockfish
    #collect_data_from_csv(csv_filename, stockfish_path)
    KfordRShuffle = False
    if KfordRShuffle:
        seeds = [42, 123, 456, 789, 101112]
        for seed in seeds:
            print(f"\n=== Esecuzione con seed {seed} ===")
            train_model_with_shuffling_and_cv(random_seed=seed, k_folds=5, epochs=5)
    else:
        train_model_from_data()
