from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

from PySide6.QtCore import Qt, QRectF, QPoint, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import QWidget

from go.board import Board, Move
from go.constants import Stone

@dataclass
class HitTest:
    x: int
    y: int
    ok: bool

class GoBoardWidget(QWidget):
    clicked = Signal(int, int)  # ✅ (x,y) emit

    def __init__(self, board: Board, parent=None):
        super().__init__(parent)
        self.board = board
        self.margin = 28
        self.cell = 28
        self.star_radius = 3
        self.stone_radius = 12

        self.hover: Optional[Tuple[int, int]] = None
        self.setMouseTracking(True)     

        self.suggestions: list[tuple[int,int,int]] = []  # (x,y,rank) rank=1..k

    def set_suggestions(self, coords_with_rank: list[tuple[int,int,int]]):
        self.suggestions = coords_with_rank
        self.update()


    def set_board(self, board: Board):
        self.board = board
        self.update()

    def minimumSizeHint(self):
        s = self.margin * 2 + self.cell * (self.board.size - 1)
        return super().minimumSizeHint().expandedTo(self.sizeHint()).expandedTo(self.rect().size())

    def sizeHint(self):
        s = self.margin * 2 + self.cell * (self.board.size - 1)
        return self.rect().size() if self.rect().size().isValid() else super().sizeHint()

    def _board_pixel_size(self) -> int:
        return self.margin * 2 + self.cell * (self.board.size - 1)

    def _grid_origin(self) -> QPoint:
        # center inside widget
        bw = self._board_pixel_size()
        ox = (self.width() - bw) // 2 + self.margin
        oy = (self.height() - bw) // 2 + self.margin
        return QPoint(ox, oy)

    def _coord_to_point(self, x: int, y: int) -> QPoint:
        o = self._grid_origin()
        return QPoint(o.x() + x * self.cell, o.y() + y * self.cell)

    def _point_to_coord(self, px: int, py: int) -> HitTest:
        o = self._grid_origin()
        fx = (px - o.x()) / self.cell
        fy = (py - o.y()) / self.cell
        x = int(round(fx))
        y = int(round(fy))
        if 0 <= x < self.board.size and 0 <= y < self.board.size:
            p = self._coord_to_point(x, y)
            # click tolerance
            if abs(px - p.x()) <= self.cell * 0.45 and abs(py - p.y()) <= self.cell * 0.45:
                return HitTest(x, y, True)
        return HitTest(-1, -1, False)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background (wood-ish 느낌: 색 지정 안 한다고 했지만 UI는 괜찮지?)
        # 색 지정 없이 시스템 기본 브러시를 쓰면 보기 안 좋아서, 최소한만:
        p.fillRect(self.rect(), QColor("#2B2B2B"))  # 다크 그레이
        
        
        # board area
        bw = self._board_pixel_size()
        o = self._grid_origin()
        board_rect = QRectF(o.x() - self.margin, o.y() - self.margin, bw, bw)
        # 바둑판 나무색
        board_color = QColor("#E3C28A")   # 연한 갈색
        p.fillRect(board_rect, board_color)

        # grid lines
        pen = QPen(QColor("#333333"))  # 완전 검정 X
        pen.setWidth(1)

        p.setPen(pen)

        # draw lines
        for i in range(self.board.size):
            a = self._coord_to_point(0, i)
            b = self._coord_to_point(self.board.size - 1, i)
            p.drawLine(a, b)
            a = self._coord_to_point(i, 0)
            b = self._coord_to_point(i, self.board.size - 1)
            p.drawLine(a, b)

        # star points (9x9/13x13/19x19)
        stars = self._star_points(self.board.size)
        for (sx, sy) in stars:
            c = self._coord_to_point(sx, sy)
            p.setBrush(QBrush(self.palette().text().color()))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(c.x() - self.star_radius, c.y() - self.star_radius,
                                 self.star_radius * 2, self.star_radius * 2))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # stones
        for y in range(self.board.size):
            for x in range(self.board.size):
                s = self.board.get(x, y)
                if s == Stone.EMPTY:
                    continue
                c = self._coord_to_point(x, y)
                r = self.stone_radius
                rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
                if s == Stone.BLACK:
                    p.setBrush(QBrush(Qt.black))
                    p.setPen(QPen(Qt.black))
                else:
                    p.setBrush(QBrush(Qt.white))
                    p.setPen(QPen(Qt.black))
                p.drawEllipse(rect)

        # hover preview (legal move only)
        if self.hover and not self.board.game_over:
            hx, hy = self.hover
            if self.board.in_bounds(hx, hy) and self.board.get(hx, hy) == Stone.EMPTY:
                m = Move(hx, hy)
                if self.board.is_legal(m):
                    c = self._coord_to_point(hx, hy)
                    r = self.stone_radius
                    rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
                    p.setPen(QPen(self.palette().highlight().color(), 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(rect)

        # suggestions overlay (rank numbers)
        if self.suggestions:
            p.setPen(QPen(QColor("#C00000"), 2))  # 빨간색 테두리
            for (sx, sy, rank) in self.suggestions:
                c = self._coord_to_point(sx, sy)
                r = int(self.stone_radius * 0.9)
                rect = QRectF(c.x() - r, c.y() - r, 2*r, 2*r)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(rect)
                p.drawText(rect, Qt.AlignCenter, str(rank))


        p.end()

    def mouseMoveEvent(self, event):
        hit = self._point_to_coord(event.position().x(), event.position().y())
        self.hover = (hit.x, hit.y) if hit.ok else None
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        hit = self._point_to_coord(event.position().x(), event.position().y())
        if hit.ok:
            self.clicked.emit(hit.x, hit.y)  # ✅ signal emit


    def _star_points(self, size: int) -> List[Tuple[int, int]]:
        if size == 19:
            pts = [3, 9, 15]
            return [(x, y) for x in pts for y in pts]
        if size == 13:
            pts = [3, 6, 9]
            return [(x, y) for x in pts for y in pts]
        if size == 9:
            pts = [2, 4, 6]
            return [(x, y) for x in pts for y in pts]
        return []
