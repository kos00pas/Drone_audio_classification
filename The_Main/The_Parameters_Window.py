import tkinter as tk

class Parameters_Window:
    def __init__(self, database, space):
        self.DATA = database
        (self.noise_canvas, self.MIN_NN, self.AGCONOFF, self.HPFONOFF, self.STATNOISEONOFF,
         self.GAMMA_NN, self.MIN_NS, self.GAMMA_NS, self.NONSTATNOISEONOFF, self.CNIONOFF,
         self.filter_canvas, self.gain_canvas, self.AGCGAIN, self.GAMMA_ENL, self.GAMMA_ETAIL,
         self.GAMMA_E, self.NLATTENONOFF, self.TRANSIENTONOFF, self.ECHOONOFF) = (None,) * 19

        self.gain(5, space)
        self.filters(15, space)
        self.noise(1, space)
        self.echo(10, space)
        self.aec(14, space)
        # Make the grid cells expand evenly
        space.grid_rowconfigure(0, weight=1)
        space.grid_columnconfigure(0, weight=1)


    def aec(self, roww, space):
        aec_label = tk.Label(space, text="AEC")
        aec_label.grid(row=roww, column=0, padx=5, pady=5)

        self.AECFREEZEONOFF = tk.Button(space, text="AECFREEZEONOFF", command=lambda: self.on_off('AECFREEZEONOFF', self.AECFREEZEONOFF))
        self.AECFREEZEONOFF.grid(row=roww, column=1, padx=5, pady=5)
        self.set_red_green('AECFREEZEONOFF', self.AECFREEZEONOFF)

        self.aec_norm = tk.DoubleVar()
        self.create_scale(space, 0.25, 16, 0.75, self.GAMMA_E, roww, 2, 'AECNORM', textt="AECNORM")

    def echo(self, roww, space):
        echo_label = tk.Label(space, text="Echo")
        echo_label.grid(row=roww, column=0, padx=5, pady=5)

        self.GAMMA_E = tk.DoubleVar()
        self.create_scale(space, 0, 3, 0.5, self.GAMMA_E, roww, 2, 'GAMMA_E', textt="GAMMA_E")

        self.GAMMA_ETAIL = tk.DoubleVar()
        self.create_scale(space, 0, 3, 0.5, self.GAMMA_ETAIL, roww, 6, 'GAMMA_ETAIL', textt="GAMMA_ETAIL")

        self.GAMMA_ENL = tk.DoubleVar()
        self.create_scale(space, 0, 5, 0.7, self.GAMMA_ENL, roww+1, 2, 'GAMMA_ENL', textt="GAMMA_ENL")

        self.ECHOONOFF = tk.Button(space, text="ECHOONOFF", command=lambda: self.on_off('ECHOONOFF', self.ECHOONOFF))
        self.ECHOONOFF.grid(row=roww, column=1, padx=5, pady=5)
        self.set_red_green('ECHOONOFF', self.ECHOONOFF)

        self.TRANSIENTONOFF = tk.Button(space, text="TRANSIENTONOFF", command=lambda: self.on_off('TRANSIENTONOFF', self.TRANSIENTONOFF))
        self.TRANSIENTONOFF.grid(row=roww+1, column=1, padx=5, pady=5)
        self.set_red_green('TRANSIENTONOFF', self.TRANSIENTONOFF)

        self.NLATTENONOFF = tk.Button(space, text="NLATTENONOFF", command=lambda: self.on_off('NLATTENONOFF', self.NLATTENONOFF))
        self.NLATTENONOFF.grid(row=roww+2, column=1, padx=5, pady=5)
        self.set_red_green('NLATTENONOFF', self.NLATTENONOFF)

        self.echo_canvas = tk.Canvas(space, height=1, bg='black')
        self.echo_canvas.grid(row=roww + 3, column=0, columnspan=13, sticky='ew', padx=5, pady=5)

    def gain(self, roww, space):
        gain_l = tk.Label(space, text="Gain")
        gain_l.grid(row=roww, column=0, padx=5, pady=5)

        self.AGCONOFF = tk.Button(space, text="AGCONOFF", command=lambda: self.on_off('AGCONOFF', self.AGCONOFF))
        self.AGCONOFF.grid(row=roww, column=1, padx=5, pady=5)
        self.set_red_green('AGCONOFF', self.AGCONOFF)

        self.AGCGAIN = tk.DoubleVar()
        self.create_scale(space, 1, 31, 1.5, self.AGCGAIN, roww+1, 2, 'AGCGAIN', textt="AGCGAIN")

        self.AGCTIME = tk.DoubleVar()
        self.create_scale(space, 0.1, 1, 0.1, self.AGCTIME, roww, 2, 'AGCTIME', textt="AGCTIME")

        self.AGCDESIREDLEVEL = tk.DoubleVar()
        self.create_scale(space, 1e-08, 0.99, 0.1, self.AGCDESIREDLEVEL, roww, 6, 'AGCDESIREDLEVEL', textt="AGCDESIREDLEVEL")

        self.gain_canvas = tk.Canvas(space, height=1, bg='black')
        self.gain_canvas.grid(row=roww + 2, column=0, columnspan=13, sticky='ew', padx=5, pady=5)

    def noise(self, roww, space):
        self.above = tk.Canvas(space, height=1, bg='black')
        self.above.grid(row=roww-1, column=0, columnspan=13, sticky='ew', padx=5, pady=5)
        noise_label = tk.Label(space, text="Noise")
        noise_label.grid(row=roww, column=0, padx=5, pady=5)

        self.CNIONOFF = tk.Button(space, text="CNIONOFF", command=lambda: self.on_off('CNIONOFF', self.CNIONOFF))
        self.CNIONOFF.grid(row=roww, column=1, padx=5, pady=5)
        self.set_red_green('CNIONOFF', self.CNIONOFF)

        self.STATNOISEONOFF = tk.Button(space, text="STATNOISEONOFF", command=lambda: self.on_off('STATNOISEONOFF', self.STATNOISEONOFF))
        self.STATNOISEONOFF.grid(row=roww+1, column=1, padx=5, pady=5)
        self.set_red_green('STATNOISEONOFF', self.STATNOISEONOFF)

        self.NONSTATNOISEONOFF = tk.Button(space, text="NONSTATNOISEONOFF", command=lambda: self.on_off('NONSTATNOISEONOFF', self.NONSTATNOISEONOFF))
        self.NONSTATNOISEONOFF.grid(row=roww+2, column=1, padx=5, pady=5)
        self.set_red_green('NONSTATNOISEONOFF', self.NONSTATNOISEONOFF)

        self.GAMMA_NS = tk.DoubleVar()
        self.create_scale(space, 0, 3, 0.5, self.GAMMA_NS, roww, 2, 'GAMMA_NS', textt="GAMMA_NS")

        self.MIN_NS = tk.DoubleVar()
        self.create_scale(space, 0, 1, 0.1, self.MIN_NS, roww, 6, 'MIN_NS', textt="MIN_NS")

        self.GAMMA_NN = tk.DoubleVar()
        self.create_scale(space, 0, 3, 0.5, self.GAMMA_NN, roww+1, 2, 'GAMMA_NN', textt="GAMMA_NN")

        self.MIN_NN = tk.DoubleVar()
        self.create_scale(space, 0, 1, 0.1, self.MIN_NN, roww+1, 6, 'MIN_NN', textt="MIN_NN")

        self.noise_canvas = tk.Canvas(space, height=1, bg='black')
        self.noise_canvas.grid(row=roww+3, column=0, columnspan=13, sticky='ew', padx=5, pady=5)

    def filters(self, roww, space):
        filters_label = tk.Label(space, text="filters")
        filters_label.grid(row=roww, column=0, padx=5, pady=5)
        self.HPFONOFF = tk.DoubleVar()
        self.create_scale(space, 0, 3, 1, self.HPFONOFF, roww, 2, 'HPFONOFF', textt="HPFONOFF")
        self.filter_canvas = tk.Canvas(space, height=1, bg='black')
        self.filter_canvas.grid(row=roww + 1, column=0, columnspan=13, sticky='ew', padx=5, pady=5)

    def on_off(self, the_param, variable):
        state = self.DATA.resp4.read_param(the_Self=self, param_string=the_param)
        if state == 0:
            self.DATA.resp4.write_param(the_param, 1)
            print(the_param, " is now ON")
            variable.config(bg='green')
        else:
            self.DATA.resp4.write_param(the_param, 0)
            print(the_param, " is now OFF")
            variable.config(bg='red')

    def set_red_green(self, the_param, the_variable):
        state = self.DATA.resp4.read_param(the_Self=self, param_string=the_param)
        if state == 0:
            the_variable.config(bg='red')
        else:
            the_variable.config(bg='green')

    def create_scale(self, space, from_val, to_val, resolution, variable, row, col, param, textt):
        def on_scale_change(val):
            value = float(val)
            self.DATA.resp4.write_param(param, value)
            print(f"{param} value changed to:", value)

        label = tk.Label(space, text=f"{textt}")
        label.grid(row=row, column=col, padx=5, pady=5)

        scale = tk.Scale(space, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL,
                         variable=variable, command=on_scale_change)
        scale.grid(row=row, column=col + 1, padx=5, pady=5)

        # Set initial value in scale
        initial_value = self.DATA.resp4.read_param(the_Self=self, param_string=param)
        variable.set(initial_value)
        scale.set(initial_value)
