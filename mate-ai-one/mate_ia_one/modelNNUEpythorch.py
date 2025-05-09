import os
import re
from collections import defaultdict
from functools import lru_cache
import chess
import csv
import numpy as np
from chess.pgn import read_game
from io import StringIO
from math import ceil
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from stockfish import Stockfish
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configurazione filtro ELO
MIN_ELO = 1800
MAX_ELO_DIFF = 100

# Configurazione campionamento
SAMPLE_EVERY_N_MOVES = 5
TARGET_SAMPLES = 5000

# Definizione della classe NNUE con PyTorch
class NNUEP(nn.Module):
    #768 rappresentando 12 tipi di pezzi × 64 caselle x mossa al bianco nero, senza considerare posizioni relative al re.
    def __init__(self, input_size=769, hidden_sizes=[512, 32, 32], dropout_rate=0.3):
        super(NNUEP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = 1
        self.dropout_rate = dropout_rate

        # Creazione dinamica degli strati nascosti
        layers = []
        prev_size = input_size  # Dimensione iniziale è quella dell'input
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size  # La dimensione successiva parte dall'output del layer corrente
        self.hidden_layers = nn.ModuleList(layers)  # Lista di strati nascosti
        self.output_layer = nn.Linear(prev_size, 1)  # Strato finale per l'output
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout per regolarizzazione

        # Inizializzazione dei pesi (es. Kaiming)
        for layer in self.hidden_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # Passaggio attraverso gli strati nascosti successivi
        for layer in self.hidden_layers:
            x = self.clipped_relu(layer(x))
            x = self.dropout(x)  # Applicazione del dropout

        x = self.output_layer(x)
        return x

    def clipped_relu(self, x):
        """ReLU con limite: valori tra 0 e 127."""
        return torch.clamp(x, 0, 127)

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    @staticmethod
    def load(filename, input_size=769, hidden_sizes=[512, 32, 32], dropout_rate=0.3):
        model = NNUEP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model

    @staticmethod
    def recunstract_info(path="../weight/weight9/0_0_0_5000.pth"):
        file_name = path.split("/")[-1]
        input_size_index, hidden_size_index, dropouts_index, size_index = file_name.split("_")
        print(path.split("_"))
        print(input_size_index, hidden_size_index, dropouts_index, size_index)
        sizeCSV = [15000]
        hidden_sizes = [[1024, 512, 256, 128], [768, 768, 768, 768, 128], ]
        dropouts = [0.1]
        input_sizes = [769]
        epochs = [200]
        return input_sizes[int(input_size_index)], hidden_sizes[int(hidden_size_index)], dropouts[int(dropouts_index)]

# Classe per l'addestramento con PyTorch
class NNUE_Trainer:

    def __init__(self, model):
        # Determina il device: GPU se disponibile, altrimenti CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print("Utilizzo della GPU (CUDA) disponibile.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Utilizzo della GPU (MPS) disponibile.")
        else:
            print("GPU non disponibile, utilizzo della CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # Aggiungiamo weight_decay per L2 regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def train(self, X, y, epochs=100, patience=10):
        # Converte i dati in tensori
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        # Sposta i dati sul device se necessario
        if self.device.type == 'cuda':
            X = X.to(self.device)
            y = y.to(self.device)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            print("loss:", loss.item())

            # Calcolo delle metriche
            with torch.no_grad():
                if self.device.type == 'cuda':
                    predictions = outputs.cpu().numpy()
                    y_np = y.cpu().numpy()
                else:
                    predictions = outputs.numpy()
                    y_np = y.numpy()
                mse = mean_squared_error(y_np, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_np, predictions)
                r2 = r2_score(y_np, predictions)
                mape = np.mean(np.abs((y_np - predictions) / (y_np + 1e-10))) * 100
                print(
                    f"Epoch {epoch + 1}/{epochs} - MSE: {mse:.4f} - RMSE: {rmse:.4f} - MAE: {mae:.4f} - R²: {r2:.4f} - MAPE: {mape:.2f}%")

            # Early stopping: controlla se il loss corrente non migliora
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping attivato dopo {epoch + 1} epoche!")
                    break

    def evaluate(self, X, y, phase="test"):
        self.model.eval()
        with torch.no_grad():
            if self.device.type == 'cuda':
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            outputs = self.model(X)
            if self.device.type == 'cuda':
                predictions = outputs.cpu().numpy()
                y_np = y.cpu().numpy()
            else:
                predictions = outputs.numpy()
                y_np = y.numpy()
            mse = mean_squared_error(y_np, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_np, predictions)
            r2 = r2_score(y_np, predictions)
            mape = np.mean(np.abs((y_np - predictions) / (y_np + 1e-10))) * 100
            print(f"{phase.capitalize()} Metrics - MSE: {mse:.4f} - RMSE: {rmse:.4f} - MAE: {mae:.4f} - R²: {r2:.4f} - MAPE: {mape:.2f}%")
            return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def fen_to_halfka_vector(fen, side=chess.WHITE):
    """
    Converte una posizione FEN in un vettore HalfKA di dimensione 45056.
    Parametri:
        fen (str): Stringa FEN della posizione.
        side (bool): chess.WHITE o chess.BLACK per scegliere il lato.
    Ritorna:
        np.array: Vettore di input di dimensione 45056.
    """
    board = chess.Board(fen)
    input_vector = np.zeros(45056, dtype=np.float32)
    king_square = board.king(side)
    if king_square is None:
        return input_vector  # Se il re non è presente, ritorna vettore zero

    # Tipi di pezzi escluso il re
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for piece_type in piece_types:
        for square in board.pieces(piece_type, side):
            # Posizione relativa al re (offset tra -63 e +63)
            rel_square = square - king_square + 63  # Sposta in range 0-126
            # Indice univoco: piece_type * 64 * 127 + rel_square * 64 + square
            idx = (piece_type - 1) * 64 * 127 + rel_square * 64 + square
            if 0 <= idx < 45056:
                input_vector[idx] = 1.0
    return input_vector

def fen_to_halfkp_vector(fen):
    """
    Converte una posizione FEN in un vettore HalfKP di dimensione 82048.
    Parametri:
        fen (str): Stringa FEN della posizione.
    Ritorna:
        np.array: Vettore di input di dimensione 82048.
    """
    board = chess.Board(fen)
    input_vector = np.zeros(82048, dtype=np.float32)
    king_white = board.king(chess.WHITE)
    king_black = board.king(chess.BLACK)
    if king_white is None or king_black is None:
        return input_vector  # Se uno dei re non è presente, vettore zero

    # Tipi di pezzi escluso il re
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    # Codifica per il bianco (prime 41024 feature)
    for piece_type in piece_types:
        for square in board.pieces(piece_type, chess.WHITE):
            rel_to_king = square - king_white + 63  # Range 0-126
            idx = (piece_type - 1) * 64 * 127 + rel_to_king * 64 + square
            if 0 <= idx < 41024:
                input_vector[idx] = 1.0

    # Codifica per il nero (seconde 41024 feature)
    for piece_type in piece_types:
        for square in board.pieces(piece_type, chess.BLACK):
            rel_to_king = square - king_black + 63  # Range 0-126
            idx = 41024 + (piece_type - 1) * 64 * 127 + rel_to_king * 64 + square
            if 41024 <= idx < 82048:
                input_vector[idx] = 1.0

    return input_vector

# Funzione per convertire FEN in vettore di input
def fen_to_input_vector(fen, input_size=769):
    board = chess.Board(fen)
    if input_size == 769:
        # Codifica attuale: 12 pezzi × 64 caselle
        input_vector = np.zeros(769)
        piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        for square in range(64):
            piece = board.piece_at(square)
            if piece is not None:
                piece_char = piece.symbol()
                feature_index = piece_to_index[piece_char] * 64 + square
                input_vector[feature_index] = 1
        input_vector[768]= 1 if board.turn == chess.WHITE else 0
    elif input_size == 768:
        # Codifica attuale: 12 pezzi × 64 caselle
        input_vector = np.zeros(768)
        piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        for square in range(64):
            piece = board.piece_at(square)
            if piece is not None:
                piece_char = piece.symbol()
                feature_index = piece_to_index[piece_char] * 64 + square
                input_vector[feature_index] = 1
    elif input_size == 1536:
        # Esempio: aggiungi features extra, come pezzi rispetto al re
        input_vector = np.zeros(1536)
        piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        king_square = board.king(board.turn)
        for square in range(64):
            piece = board.piece_at(square)
            if piece is not None:
                piece_char = piece.symbol()
                # Prima parte: posizione assoluta (come per 768)
                feature_index = piece_to_index[piece_char] * 64 + square
                input_vector[feature_index] = 1
                # Seconda parte: posizione relativa al re
                rel_index = piece_to_index[piece_char] * 64 + (square - king_square + 64) % 64
                input_vector[768 + rel_index] = 1
    elif input_size == 45056:
        input_vector = fen_to_halfka_vector(fen)
    elif input_size == 82048:
        input_vector = fen_to_halfkp_vector(fen)
    else:
        raise ValueError(f"Input size {input_size} non supportato")
    return input_vector

# Funzioni di parsing e campionamento (invariate)
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

def sample_positions(positions,max_rows=TARGET_SAMPLES):
    if max_rows is None:
        max_rows = len(positions)
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
        target = ceil(max_rows * (len(items) / total))
        target = min(target, len(items))
        indices = np.arange(len(items))
        try:
            selected_indices = np.random.choice(indices, size=target, replace=(target > len(items)))
            sampled += [items[i] for i in selected_indices]
        except Exception as e:
            print(f"Errore campionamento {phase}: {str(e)}")
            continue
    try:
        return np.random.permutation(sampled)[:max_rows].tolist()
    except:
        return sampled[:max_rows]

def collect_data_from_csv_stockfish(filename, stockfish_path):
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
                white_elo = int(row['WhiteElo'])
                black_elo = int(row['BlackElo'])
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
                        if (i<10 or i % SAMPLE_EVERY_N_MOVES == 0 or is_critical_position(board) or board.is_game_over()):
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

def append_to_pickle(new_data, filename='chess_data.pickle'):
    # Se il file esiste, carica i dati esistenti, altrimenti usa una lista vuota
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            try:
                existing_data = pickle.load(f)
            except Exception as e:
                print(f"Errore nel caricamento dei dati esistenti: {e}")
                existing_data = []
    else:
        existing_data = []

    # Unisci i dati esistenti con quelli nuovi
    updated_data = existing_data + new_data

    # Salva il risultato in modalità scrittura binaria (sovrascrive il file)
    with open(filename, 'wb') as f:
        pickle.dump(updated_data, f)
    print(f"File '{filename}' aggiornato con {len(new_data)} nuovi elementi.")

def collect_data_from_csv(filename:str,datapickename:str="chess_data.pickle",max_rows:int=None):
    """
    Legge un CSV contenente posizioni in FEN e valutazioni, supportando tre formati:
      - chessData.csv: colonne "FEN,Evaluation"
      - randomeval.csv: colonne "FEN,Evaluation"
      - taticseval.csv: colonne "FEN,Evaluation,Move" (la colonna Move viene ignorata)

    Le valutazioni sono in centi-pawn (o in formato mate, es. "#+2") e vengono convertite in un valore normalizzato
    nell'intervallo [-1, 1]. I dati finali vengono bilanciati con sample_positions e salvati in "chess_data.pickle",
    in modo che la funzione train_model_from_data possa utilizzarli senza ulteriori modifiche.
    """
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Controlla che le colonne essenziali siano presenti
        if "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
            raise ValueError("Il file CSV deve contenere almeno le colonne 'FEN' e 'Evaluation'")
        row_count = 0
        for row in reader:
            # Se è stato impostato un limite, esce dal ciclo una volta raggiunto
            if max_rows is not None and row_count >= max_rows:
                break

            try:
                fen = row["FEN"].strip()
                eval_str = row["Evaluation"].strip()
                # Gestione della valutazione mate: se la stringa inizia con "#" viene interpretata come valore estremo
                if eval_str.startswith("#"):
                    evaluation = 1.0 if eval_str[1] == '+' else -1.0
                else:
                    # Rimuove eventuali segni '+' e converte in float dividendo per 1000
                    eval_clean = eval_str.replace('+', '')
                    evaluation = float(eval_clean) / 1000.0
                    evaluation = max(min(evaluation, 1.0), -1.0)
                data.append((fen, evaluation))
            except Exception as e:
                print(f"Errore processing row: {row} - {e}")
                continue

            row_count += 1

    # Esegui il campionamento bilanciato usando la funzione sample_positions
    balanced_data = sample_positions(positions=data,max_rows=max_rows)
    # Salva i dati in un file pickle per il training successivo

    append_to_pickle(balanced_data,datapickename)

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

# Funzione per addestrare il modello
def train_model_from_data(datapicke="chess_data.pickle",filename="nnuep_model.pth",reportname="../report/training/training_report_p.txt'",input_size=769, hidden_sizes=[512, 32, 32], dropout_rate=0.3,epoch=100):
    with open(datapicke, 'rb') as f:
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

    X = [fen_to_input_vector(fen, input_size=input_size) for fen, _ in clean_data]
    y = [eval_val for _, eval_val in clean_data]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Se esiste già il file dei pesi, carica il modello, altrimenti creane uno nuovo.
    if os.path.exists(filename):
        print("Carico pesi esistenti da 'nnuep_model.pth'...")
        model = NNUEP.load(filename,input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    else:
        print("Nessun file di pesi trovato. Creo un nuovo modello NNUEP.")
        model = NNUEP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    trainer = NNUE_Trainer(model)

    print("\nAvvio training...")
    trainer.train(X_train, y_train, epochs=epoch)

    print("\nValutazione sul set di training...")
    train_eval_metrics = trainer.evaluate(X_train, y_train, phase="train")
    print("\nValutazione sul set di test...")
    test_metrics = trainer.evaluate(X_test, y_test, phase="test")

    # Salva i pesi aggiornati al termine del training.
    model.save_weights(filename)

    # Generazione report (opzionale)
    with open(reportname, 'w') as f:
        f.write("=== Report di Addestramento e Valutazione ===\n\n")
        f.write("Dataset:\n")
        f.write(f" - Dimensioni totali: {len(X)}\n")
        f.write(f" - Training set: {len(X_train)} ({len(X_train) / len(X):.1%})\n")
        f.write(f" - Test set: {len(X_test)} ({len(X_test) / len(X):.1%})\n\n")

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

# Funzione per addestrare il modello con shuffling e cross-validation
def train_model_with_shuffling_and_cv(random_seed=42, k_folds=5, epochs=100):
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
    X = [fen_to_input_vector(fen,input_size=769) for fen, _ in clean_data]
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
        model = NNUEP()
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
    report_path = f'../report/training/training_report_nnuep_seed_{random_seed}.txt'
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


def train_model_with_kfold_cv(datapicke="chess_data.pickle", filename="nnuep_model.pth",
                              reportname="../report/training/training_report_cv.txt",
                              input_size=769, hidden_sizes=[512, 32, 32], dropout_rate=0.3, epoch=100, random_seed=42,
                              k_folds=5):
    """
    Addestra il modello con k-fold cross-validation e shuffling casuale, restituendo metriche dettagliate per grafici.

    Parametri:
    - datapicke: Percorso del file pickle con i dati.
    - filename: Percorso per salvare i pesi del modello.
    - reportname: Percorso per salvare il report.
    - input_size: Dimensione dell'input della rete (default: 769).
    - hidden_sizes: Lista delle dimensioni degli strati nascosti (default: [512, 32, 32]).
    - dropout_rate: Tasso di dropout (default: 0.3).
    - epoch: Numero di epoche per l'addestramento (default: 100).
    - random_seed: Seme per lo shuffling casuale (default: 42).
    - k_folds: Numero di fold per la cross-validation (default: 5).

    Ritorna:
    - Dizionario con metriche dettagliate per ogni fold e medie sui fold.
    """
    # Caricamento e pulizia dei dati
    with open(datapicke, 'rb') as f:
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

    X = [fen_to_input_vector(fen, input_size=input_size) for fen, _ in clean_data]
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

    # Struttura per salvare i dati per i grafici
    cv_results = {
        'folds': [],
        'parameters': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'epochs': epoch,
            'dataset_size': len(X),
            'random_seed': random_seed,
            'k_folds': k_folds
        }
    }

    fold_metrics = []
    patience = 10  # Early stopping patience

    # Esecuzione della cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Crea un nuovo modello per ogni fold
        model = NNUEP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
        trainer = NNUE_Trainer(model)

        # Struttura per il training history del fold corrente
        fold_history = {
            'epochs': [],
            'train_r2': [],
            'train_loss': [],
            'test_r2': []
        }

        # Preparazione dei tensori
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(trainer.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(trainer.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(trainer.device)

        best_loss = float('inf')
        epochs_no_improve = 0

        print("Avvio training...")
        for e in range(epoch):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            outputs = trainer.model(X_train_tensor)
            loss = trainer.criterion(outputs, y_train_tensor)
            loss.backward()
            trainer.optimizer.step()

            # Calcolo metriche sul training set
            with torch.no_grad():
                if trainer.device.type == 'cuda':
                    train_pred = outputs.cpu().numpy()
                    y_train_np = y_train_tensor.cpu().numpy()
                    test_pred = trainer.model(X_test_tensor).cpu().numpy()
                    y_test_np = y_test_tensor.cpu().numpy()
                else:
                    train_pred = outputs.numpy()
                    y_train_np = y_train_tensor.numpy()
                    test_pred = trainer.model(X_test_tensor).numpy()
                    y_test_np = y_test_tensor.numpy()
                train_r2 = r2_score(y_train_np, train_pred)
                test_r2 = r2_score(y_test_np, test_pred)

            # Salva i dati per il grafico
            fold_history['epochs'].append(e + 1)
            fold_history['train_r2'].append(train_r2)
            fold_history['train_loss'].append(loss.item())
            fold_history['test_r2'].append(test_r2)

            print(
                f"Epoch {e + 1}/{epoch} - Loss: {loss.item():.4f} - Train R²: {train_r2:.4f} - Test R²: {test_r2:.4f}")

            # Early stopping
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping attivato dopo {e + 1} epoche!")
                    break

        # Valutazione finale sul fold
        print("Valutazione finale sul training set...")
        train_metrics = trainer.evaluate(X_train, y_train, phase="train")
        print("Valutazione finale sul test set...")
        test_metrics = trainer.evaluate(X_test, y_test, phase="test")

        fold_data = {
            'fold_number': fold + 1,
            'history': fold_history,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        cv_results['folds'].append(fold_data)
        fold_metrics.append(test_metrics)

    # Calcolo delle metriche medie sui fold
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics]),
        'mape': np.mean([m['mape'] for m in fold_metrics])
    }
    cv_results['avg_metrics'] = avg_metrics

    # Salva i pesi dell'ultimo modello (opzionale)
    model.save_weights(filename)

    # Generazione report
    with open(reportname, 'w') as f:
        f.write("=== Report di K-Fold Cross-Validation ===\n\n")
        f.write(f"Random Seed: {random_seed}\n")
        f.write(f"Numero di Fold: {k_folds}\n")
        f.write("Dataset:\n")
        f.write(f" - Dimensioni totali: {len(X)}\n\n")
        for fold_data in cv_results['folds']:
            f.write(f"Fold {fold_data['fold_number']}:\n")
            f.write("  Metriche di Valutazione (Training Set):\n")
            f.write(f"    - MSE: {fold_data['train_metrics']['mse']:.4f}\n")
            f.write(f"    - RMSE: {fold_data['train_metrics']['rmse']:.4f}\n")
            f.write(f"    - MAE: {fold_data['train_metrics']['mae']:.4f}\n")
            f.write(f"    - R²: {fold_data['train_metrics']['r2']:.4f}\n")
            f.write(f"    - MAPE: {fold_data['train_metrics']['mape']:.2f}%\n")
            f.write("  Metriche di Valutazione (Test Set):\n")
            f.write(f"    - MSE: {fold_data['test_metrics']['mse']:.4f}\n")
            f.write(f"    - RMSE: {fold_data['test_metrics']['rmse']:.4f}\n")
            f.write(f"    - MAE: {fold_data['test_metrics']['mae']:.4f}\n")
            f.write(f"    - R²: {fold_data['test_metrics']['r2']:.4f}\n")
            f.write(f"    - MAPE: {fold_data['test_metrics']['mape']:.2f}%\n\n")
        f.write("Metriche Medie su Tutti i Fold (Test Set):\n")
        f.write(f" - MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f" - RMSE: {avg_metrics['rmse']:.4f}\n")
        f.write(f" - MAE: {avg_metrics['mae']:.4f}\n")
        f.write(f" - R²: {avg_metrics['r2']:.4f}\n")
        f.write(f" - MAPE: {avg_metrics['mape']:.2f}%\n")

    print(f"\nReport salvato in '{reportname}'")
    return cv_results

if __name__ == "__main__":



    csv_filename = '../dataset/chess_games.csv'  # Percorso fisso per il file CSV
    stockfish_path = '/opt/homebrew/Cellar/stockfish/17/bin/stockfish'  # Percorso fisso per Stockfish
    #collect_data_from_csv_stockfish(csv_filename, stockfish_path)
    KfordRShuffle=False
    if KfordRShuffle:
        seeds = [42, 123, 456, 789, 101112]
        for seed in seeds[4:]:
            print(f"\n=== Esecuzione con seed {seed}   ===")
            train_model_with_shuffling_and_cv(random_seed=seed, k_folds=5, epochs=80)
    else:
        train_model_from_data(datapicke="./datapickle/chess_data_1500000.pickle",filename="./weight/nnuep_model.pth",input_size=769,hidden_sizes=[769,384,192,96],dropout_rate=0.1,epoch=100)