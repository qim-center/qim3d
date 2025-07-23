import gradio as gr

LIGHT_BLUE = '#60a5fa'
DARK_BLUE = "#2472d2"
RUN_COLOR = '#198754'
BRIGHT_RUN_COLOR = '#299764'
CANCEL_COLOR = '#dc3545'

class QimTheme(gr.themes.Default):

    """
    Theme for qim3d gradio interfaces.
    The theming options are quite broad. However if there is something you can not achieve with this theme
    there is a possibility to add some more css if you override _get_css_theme function as shown at the bottom
    in comments.
    """

    def __init__(self, force_light_mode: bool = True):
        """
        Parameters
        ----------
        - force_light_mode (bool, optional): Gradio themes have dark mode by default.
                QIM platform is not ready for dark mode yet, thus the tools should also be in light mode.
                This sets the darkmode values to be the same as light mode values.

        """
        super().__init__()
        self.force_light_mode = force_light_mode
        self.general_values()  # Not color related
        self.set_light_mode_values()
        self.set_dark_mode_values()  # Checks the light mode setting inside

    def general_values(self):
        self.set_data_explorer()
        self.set_button()
        self.set_h1()

    def set_light_mode_values(self):
        self.set_light_primary_button()
        self.set_light_secondary_button()
        self.set_light_checkbox()
        self.set_light_cancel_button()
        self.set_light_example()
        self.set_light_slider()

    def set_dark_mode_values(self):
        if self.force_light_mode:
            for attr in [
                dark_attr
                for dark_attr in dir(self)
                if not dark_attr.startswith('_') and dark_attr.endswith('dark')
            ]:
                self.__dict__[attr] = self.__dict__[
                    attr[:-5]
                ]  # ligth and dark attributes have same names except for '_dark' at the end
        else:
            self.set_dark_primary_button()
            # Secondary button looks good by default in dark mode
            self.set_dark_checkbox()
            self.set_dark_cancel_button()
            # Example looks good by default in dark mode
            self.set_dark_slider()

    def set_button(self):
        self.button_transition = '0.15s'
        self.button_large_text_weight = 'normal'
        self.button_border_width = '1px'
    
    def set_data_explorer(self):
        # Changes the color of the arrow
        self.color_accent = LIGHT_BLUE

    def set_light_primary_button(self):
        self.button_primary_background_fill = '#FFFFFF'
        self.button_primary_background_fill_hover = RUN_COLOR
        self.button_primary_border_color = RUN_COLOR
        self.button_primary_border_color_hover = RUN_COLOR
        self.button_primary_text_color = RUN_COLOR
        self.button_primary_text_color_hover = '#FFFFFF'

    def set_dark_primary_button(self):
        self.button_primary_background_fill_dark = (
            self.button_primary_background_fill_hover
        )
        self.button_primary_background_fill_hover_dark = BRIGHT_RUN_COLOR
        self.button_primary_border_color_dark = self.button_primary_border_color
        self.button_primary_border_color_hover_dark = BRIGHT_RUN_COLOR

    def set_light_secondary_button(self):
        self.button_secondary_background_fill = 'white'

    def set_light_example(self):
        """
        This sets how the examples in gradio.Examples look like. Used in iso3d.
        """
        self.border_color_accent = self.neutral_100
        self.color_accent_soft = self.neutral_100

    def set_h1(self):
        self.text_xxl = '2.5rem'

    def set_light_checkbox(self):
        self.checkbox_background_color_selected = LIGHT_BLUE
        self.checkbox_border_color_selected = LIGHT_BLUE
        self.checkbox_border_color_focus = LIGHT_BLUE

    def set_dark_checkbox(self):
        self.checkbox_border_color_dark = self.neutral_500
        self.checkbox_border_color_focus_dark = DARK_BLUE
        self.checkbox_border_color_selected_dark = DARK_BLUE
        self.checkbox_background_color_selected_dark = DARK_BLUE
        self.checkbox_background_color_selected_dark = DARK_BLUE

    def set_light_cancel_button(self):
        self.button_cancel_background_fill = 'white'
        self.button_cancel_background_fill_hover = CANCEL_COLOR
        self.button_cancel_border_color = CANCEL_COLOR
        self.button_cancel_text_color = CANCEL_COLOR
        self.button_cancel_text_color_hover = 'white'

    def set_dark_cancel_button(self):
        self.button_cancel_background_fill_dark = CANCEL_COLOR
        self.button_cancel_background_fill_hover_dark = 'red'
        self.button_cancel_border_color_dark = CANCEL_COLOR
        self.button_cancel_border_color_hover_dark = 'red'
        self.button_cancel_text_color_dark = 'white'

    def set_light_slider(self):
        self.slider_color = LIGHT_BLUE
    
    def set_dark_slider(self):
        self.slider_color_dark = DARK_BLUE



    # def _get_theme_css(self):
    #     sup = super()._get_theme_css()
    #     return "\n.svelte-182fdeq {\nbackground: rgba(255, 0, 0, 0.5) !important;\n}\n" + sup # You have to use !important, so it overrides other css
