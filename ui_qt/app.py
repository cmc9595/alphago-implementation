from __future__ import annotations
from typing import Optional, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QCheckBox, QSpinBox
)

from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QCursor


from go.board import Board, Move, IllegalMove
from go.constants import Stone, opponent
from go.scoring import chinese_area_score

from ui_qt.widgets import GoBoardWidget

from bots.random_bot import RandomBot
from bots.base import Bot
from bots.mcts_bot import MCTSBot


class AIWorker(QObject):
    finished = Signal(object)           # (Move, stats)
    progress = Signal(int, int, object) # (done, total, stats)
    error = Signal(str)

    def __init__(self, bot: Bot, board: Board):
        super().__init__()
        self.bot = bot
        self.board = board  # 이미 copy된 보드

    @Slot()
    def run(self):
        try:
            if hasattr(self.bot, "select_move_with_stats"):
                def cb(done: int, total: int, stats):
                    # stats는 list[MoveStat]
                    self.progress.emit(done, total, stats)

                move, stats = self.bot.select_move_with_stats(
                    self.board, topk=5, progress_cb=cb, progress_every=10
                )
            else:
                move = self.bot.select_move(self.board)
                stats = []
            self.finished.emit((move, stats))
        except Exception as e:
            self.error.emit(repr(e))




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaGo-Lee Implementation — Go UI (Engine Ready)")

        # game state
        self.board = Board(size=19, komi=7.5, superko=False)
        self.undo_stack: List = []

        # mode
        self.use_ai = False
        self.ai_color: Stone = Stone.WHITE
        # self.bot: Bot = RandomBot()
        self.bot: Bot = MCTSBot(num_simulations=200, rollout_depth=80)

        # self.bot: Bot = MCTSBot(num_simulations=400, rollout_depth=120)


        # ui
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left: board
        self.board_widget = GoBoardWidget(self.board, parent=self)
        self.board_widget.clicked.connect(self.on_board_click)  # ✅ 추가
        root.addWidget(self.board_widget, stretch=1)

        # right: controls
        panel = QWidget()
        panel.setFixedWidth(320)
        pv = QVBoxLayout(panel)
        pv.setAlignment(Qt.AlignTop)
        root.addWidget(panel)

        self.lbl_turn = QLabel()
        self.lbl_caps = QLabel()
        self.lbl_score = QLabel()
        pv.addWidget(self.lbl_turn)
        pv.addWidget(self.lbl_caps)
        pv.addWidget(self.lbl_score)

        pv.addSpacing(12)

        # settings row
        row = QHBoxLayout()
        pv.addLayout(row)
        row.addWidget(QLabel("Board"))
        self.cmb_size = QComboBox()
        self.cmb_size.addItems(["9", "13", "19"])
        self.cmb_size.setCurrentText("19")
        row.addWidget(self.cmb_size)

        row2 = QHBoxLayout()
        pv.addLayout(row2)
        row2.addWidget(QLabel("Komi"))
        self.spn_komi = QSpinBox()
        self.spn_komi.setRange(0, 15)
        self.spn_komi.setValue(8)  # 7.5는 float인데 UI 단순화. 아래에서 7.5로 설정 가능
        row2.addWidget(self.spn_komi)

        self.chk_superko = QCheckBox("Superko")
        self.chk_superko.setChecked(False)
        pv.addWidget(self.chk_superko)

        self.btn_new = QPushButton("New Game")
        self.btn_new.clicked.connect(self.on_new_game)
        pv.addWidget(self.btn_new)

        pv.addSpacing(12)

        # actions
        self.btn_pass = QPushButton("Pass")
        self.btn_pass.clicked.connect(self.on_pass)
        pv.addWidget(self.btn_pass)

        self.btn_resign = QPushButton("Resign")
        self.btn_resign.clicked.connect(self.on_resign)
        pv.addWidget(self.btn_resign)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.on_undo)
        pv.addWidget(self.btn_undo)

        pv.addSpacing(12)

        # AI controls
        self.chk_ai = QCheckBox("Human vs AI")
        self.chk_ai.setChecked(False)
        self.chk_ai.stateChanged.connect(self.on_toggle_ai)
        pv.addWidget(self.chk_ai)

        row3 = QHBoxLayout()
        pv.addLayout(row3)
        row3.addWidget(QLabel("AI Color"))
        self.cmb_ai_color = QComboBox()
        self.cmb_ai_color.addItems(["White", "Black"])
        self.cmb_ai_color.setCurrentText("White")
        self.cmb_ai_color.currentTextChanged.connect(self.on_ai_color_changed)
        row3.addWidget(self.cmb_ai_color)

        self.btn_ai_move = QPushButton("AI Move Now")
        self.btn_ai_move.clicked.connect(self.on_ai_move_now)
        pv.addWidget(self.btn_ai_move)

        self.lbl_ai = QLabel("AI: idle")
        pv.addWidget(self.lbl_ai)

        self.lbl_topk = QLabel("TopK:\n-")
        self.lbl_topk.setWordWrap(True)
        pv.addWidget(self.lbl_topk)


        pv.addStretch(1)

        self.refresh_status()

    # ---- hooks from board widget ----
    def on_board_click(self, x: int, y: int):
        if self.board.game_over:
            return

        # if it's AI's turn, ignore clicks
        if self.use_ai and self.board.to_play == self.ai_color:
            return

        self.clear_ai_hints()
        self.try_play(Move(x, y))

        # if AI enabled and now AI to move
        if self.use_ai and (not self.board.game_over) and self.board.to_play == self.ai_color:
            self.run_ai_move()

    # ---- game ops ----
    def snapshot_push(self):
        self.undo_stack.append(self.board._snapshot_min())

    def try_play(self, move: Move):
        try:
            self.snapshot_push()
            self.board.play(move)
        except IllegalMove as e:
            self.undo_stack.pop()  # revert snapshot push
            QMessageBox.information(self, "Illegal Move", str(e))
        finally:
            self.refresh_status()
            self.board_widget.update()
            if self.board.game_over:
                self.on_game_over()

    def clear_ai_hints(self):
        self.board_widget.set_suggestions([])
        self.lbl_topk.setText("TopK:\n-")


    def on_pass(self):
        if self.board.game_over:
            return
        self.clear_ai_hints()
        self.try_play(Move.pass_move())

        if self.use_ai and (not self.board.game_over) and self.board.to_play == self.ai_color:
            self.run_ai_move()

    def on_resign(self):
        if self.board.game_over:
            return
        self.try_play(Move.resign())

    def on_undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        self.board._restore_min(snap)
        self.refresh_status()
        self.board_widget.update()
        self.clear_ai_hints()

    def on_new_game(self):
        size = int(self.cmb_size.currentText())
        # komi: UI 단순화를 위해 정수만 받았는데, 19줄 기준 komi=7.5를 기본으로 쓰고 싶으면 아래처럼:
        komi = 7.5 if self.spn_komi.value() == 8 else float(self.spn_komi.value())
        superko = self.chk_superko.isChecked()

        self.board = Board(size=size, komi=komi, superko=superko)
        self.undo_stack.clear()
        self.board_widget.set_board(self.board)
        self.refresh_status()
        self.board_widget.update()

        # 새 게임에서 AI가 선이면 즉시 두게
        if self.use_ai and self.board.to_play == self.ai_color:
            self.run_ai_move()

        self.clear_ai_hints()

    # ---- status / scoring ----
    def refresh_status(self):
        turn = "Black" if self.board.to_play == Stone.BLACK else "White"
        self.lbl_turn.setText(f"Turn: {turn}")

        self.lbl_caps.setText(
            f"Captures  B:{self.board.captures_black}  W:{self.board.captures_white}"
        )

        b, w = chinese_area_score(self.board.grid, self.board.komi)
        lead = ("Black" if b > w else "White") + f" +{abs(b - w):.1f}"
        self.lbl_score.setText(f"Chinese Area Score  B:{b:.1f}  W:{w:.1f}   ({lead})")

    def on_game_over(self):
        # if resigned, winner already set
        if self.board.winner is not None:
            winner = "Black" if self.board.winner == Stone.BLACK else "White"
            QMessageBox.information(self, "Game Over", f"Winner: {winner} (resign)")
            return

        # 2-pass end: score
        b, w = chinese_area_score(self.board.grid, self.board.komi)
        if b > w:
            msg = f"Game Over (2 passes)\nBlack wins by {b - w:.1f}"
        else:
            msg = f"Game Over (2 passes)\nWhite wins by {w - b:.1f}"
        QMessageBox.information(self, "Game Over", msg)

    # ---- AI wiring ----
    def on_toggle_ai(self, state):
        self.use_ai = (state == Qt.Checked)
        # AI 체크했는데 지금이 AI 턴이면 바로 진행
        if self.use_ai and (not self.board.game_over) and self.board.to_play == self.ai_color:
            self.run_ai_move()

    def on_ai_color_changed(self, text: str):
        self.ai_color = Stone.WHITE if text.lower().startswith("w") else Stone.BLACK
        if self.use_ai and (not self.board.game_over) and self.board.to_play == self.ai_color:
            self.run_ai_move()

    def on_ai_move_now(self):
        print("AI Move Now clicked")

        if self.board.game_over:
            return
        if self.board.to_play != self.ai_color:
            self.lbl_ai.setText("AI: not my turn")
            QMessageBox.information(self, "AI", "Not AI's turn.")
            return
        self.run_ai_move()


    def run_ai_move(self):
        # ✅ UI 안 멈추게 QThread로 실행
        self.btn_ai_move.setEnabled(False)
        self.btn_pass.setEnabled(False)
        self.btn_resign.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_new.setEnabled(False)
        self.chk_ai.setEnabled(False)
        self.cmb_ai_color.setEnabled(False)

        self.lbl_ai.setText("AI: thinking...")
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # MCTS 중에 메인 보드가 바뀌면 꼬이니까 copy해서 넘김
        board_copy = self.board.copy()

        self.ai_thread = QThread(self)
        self.ai_worker = AIWorker(self.bot, board_copy)
        self.ai_worker.moveToThread(self.ai_thread)

        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.finished.connect(self._on_ai_finished)
        self.ai_worker.error.connect(self._on_ai_error)

        # cleanup
        self.ai_worker.finished.connect(self.ai_thread.quit)
        self.ai_worker.finished.connect(self.ai_worker.deleteLater)
        self.ai_thread.finished.connect(self.ai_thread.deleteLater)

        self.ai_worker.error.connect(self.ai_thread.quit)
        self.ai_worker.error.connect(self.ai_worker.deleteLater)

        self.ai_worker.progress.connect(self._on_ai_progress)

        self.ai_thread.start()

    @Slot(int, int, object)
    def _on_ai_progress(self, done: int, total: int, stats):
        # 상태 라벨
        self.lbl_ai.setText(f"AI: thinking... {done}/{total}")

        # TopK 표시
        if stats:
            lines = []
            for i, st in enumerate(stats, 1):
                winp = (st.q + 1.0) * 0.5 * 100.0
                lines.append(f"{i}. ({st.move.x+1},{st.move.y+1})  N={st.visits}  Q={st.q:+.3f}  ~{winp:.1f}%")
            self.lbl_topk.setText("TopK:\n" + "\n".join(lines))

            # 보드 오버레이 (상위 3개)
            coords = []
            for i, st in enumerate(stats[:5], 1):
                if not st.move.is_pass and not st.move.is_resign:
                    coords.append((st.move.x, st.move.y, i))
            self.board_widget.set_suggestions(coords)
        else:
            self.lbl_topk.setText("TopK:\n-")
            self.board_widget.set_suggestions([])


    @Slot(object)
    def _on_ai_finished(self, payload):
        print("AI finished")
        
        QApplication.restoreOverrideCursor()
        self.lbl_ai.setText("AI: idle")

        move, stats = payload  # stats: list[MoveStat]

        # TopK 텍스트 표시
        if stats:
            lines = []
            for i, st in enumerate(stats, 1):
                # q는 [-1,1]인데 보기 좋게 승률 느낌으로 변환(대충)
                winp = (st.q + 1.0) * 0.5 * 100.0
                lines.append(f"{i}. ({st.move.x+1},{st.move.y+1})  N={st.visits}  Q={st.q:+.3f}  ~{winp:.1f}%")
            self.lbl_topk.setText("TopK:\n" + "\n".join(lines))

            # 보드 오버레이(상위 3개만)
            coords = []
            for i, st in enumerate(stats[:5], 1):
                if not st.move.is_pass and not st.move.is_resign:
                    coords.append((st.move.x, st.move.y, i))
            self.board_widget.set_suggestions(coords)
        else:
            self.lbl_topk.setText("TopK:\n-")
            self.board_widget.set_suggestions([])

        # 버튼 복구
        self.btn_ai_move.setEnabled(True)
        self.btn_pass.setEnabled(True)
        self.btn_resign.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.btn_new.setEnabled(True)
        self.chk_ai.setEnabled(True)
        self.cmb_ai_color.setEnabled(True)

        # 실제 보드에 착수
        self.try_play(move)


    @Slot(str)
    def _on_ai_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self.lbl_ai.setText("AI: error")

        self.btn_ai_move.setEnabled(True)
        self.btn_pass.setEnabled(True)
        self.btn_resign.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.btn_new.setEnabled(True)
        self.chk_ai.setEnabled(True)
        self.cmb_ai_color.setEnabled(True)

        QMessageBox.critical(self, "AI Error", msg)



def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1100, 820)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
