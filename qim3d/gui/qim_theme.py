import gradio as gr

class QimTheme(gr.themes.Default):
    """
    Theme for qim3d gradio interfaces.
    The theming options are quite broad. However if there is something you can not achieve with this theme
    there is a possibility to add some more css if you override _get_css_theme function as shown at the bottom
    in comments.
    """
    def __init__(self, force_light_mode: bool = True):
        """
        Parameters:
        -----------
        - force_light_mode (bool, optional): Gradio themes have dark mode by default. 
                QIM platform is not ready for dark mode yet, thus the tools should also be in light mode.
                This sets the darkmode values to be the same as light mode values.
        """
        super().__init__()
        self.force_light_mode = force_light_mode
        self.general_values() # Not color related
        self.set_light_mode_values()
        self.set_dark_mode_values() # Checks the light mode setting inside

    def general_values(self):
        self.set_button()
        self.set_h1()
        
    def set_light_mode_values(self):
        self.set_light_primary_button()
        self.set_light_secondary_button()
        self.set_light_checkbox()
        self.set_light_cancel_button()
        self.set_light_example()

    def set_dark_mode_values(self):
        if self.force_light_mode:
            for attr in [dark_attr for dark_attr in dir(self) if not dark_attr.startswith("_") and dark_attr.endswith("dark")]:
                self.__dict__[attr] = self.__dict__[attr[:-5]] # ligth and dark attributes have same names except for '_dark' at the end
        else:
            self.set_dark_primary_button()
            # Secondary button looks good by default in dark mode
            self.set_dark_checkbox()
            self.set_dark_cancel_button()
            # Example looks good by default in dark mode

    def set_button(self):
        self.button_transition = "0.15s"
        self.button_large_text_weight = "normal"

    def set_light_primary_button(self):
        self.run_color = "#198754"
        self.button_primary_background_fill = "#FFFFFF"
        self.button_primary_background_fill_hover = self.run_color
        self.button_primary_border_color = self.run_color
        self.button_primary_text_color = self.run_color
        self.button_primary_text_color_hover = "#FFFFFF"

    def set_dark_primary_button(self):
        self.bright_run_color = "#299764"
        self.button_primary_background_fill_dark = self.button_primary_background_fill_hover
        self.button_primary_background_fill_hover_dark = self.bright_run_color
        self.button_primary_border_color_dark = self.button_primary_border_color
        self.button_primary_border_color_hover_dark = self.bright_run_color

    def set_light_secondary_button(self):
        self.button_secondary_background_fill = "white"

    def set_light_example(self):
        """
        This sets how the examples in gradio.Examples look like. Used in iso3d.
        """
        self.border_color_accent = self.neutral_100
        self.color_accent_soft = self.neutral_100

    def set_h1(self):
        self.text_xxl = "2.5rem"

    def set_light_checkbox(self):
        light_blue = "#60a5fa"
        self.checkbox_background_color_selected = light_blue
        self.checkbox_border_color_selected = light_blue
        self.checkbox_border_color_focus = light_blue

    def set_dark_checkbox(self):
        self.checkbox_border_color_dark = self.neutral_500
        self.checkbox_border_color_focus_dark = self.checkbox_border_color_focus_dark

    def set_light_cancel_button(self):
        self.cancel_color = "#dc3545"
        self.button_cancel_background_fill = "white"
        self.button_cancel_background_fill_hover = self.cancel_color
        self.button_cancel_border_color = self.cancel_color
        self.button_cancel_text_color = self.cancel_color
        self.button_cancel_text_color_hover = "white"

    def set_dark_cancel_button(self):
        self.button_cancel_background_fill_dark = self.cancel_color
        self.button_cancel_background_fill_hover_dark = "red"
        self.button_cancel_border_color_dark = self.cancel_color
        self.button_cancel_border_color_hover_dark = "red"
        self.button_cancel_text_color_dark = "white"

    # def _get_theme_css(self):
    #     sup = super()._get_theme_css()
    #     return "\n.svelte-182fdeq {\nbackground: rgba(255, 0, 0, 0.5) !important;\n}\n" + sup # You have to use !important, so it overrides other css
        