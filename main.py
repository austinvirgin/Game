import arcade
import arcade.gui
from pong import Pong
import random

WIDTH = 800
HEIGHT = 600
TITLE = "PONG"

class MainView(arcade.View):

    def __init__(self):
        super().__init__()
        self.manager = arcade.gui.UIManager()

        start_game = arcade.gui.UIFlatButton(text = 'Start Game', width = 250)

        @start_game.event("on_click")
        def on_click_switch_button(event):
            menu_view = Pong(self)
            menu_view.setup()
            self.window.show_view(menu_view)

        self.anchor = self.manager.add(arcade.gui.UIAnchorLayout())

        self.anchor.add(
            anchor_x="center_x",
            anchor_y="center_y",
            child = start_game
        )

    def on_show_view(self):
        arcade.set_background_color(arcade.color.BABY_BLUE)
        self.manager.enable()

    def on_hide_view(self):
        self.manager.disable()

    def on_draw(self):
        self.clear()
        self.manager.draw()

def main():
    window = arcade.Window(WIDTH, HEIGHT, TITLE)
    main_view = MainView()
    window.show_view(main_view)
    arcade.run()

if __name__ == "__main__":
    main()