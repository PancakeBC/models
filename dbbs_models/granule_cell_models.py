import numpy as np
from patch import p
from ._shared import DbbsNeuronModel
from math import floor

class GranuleBase:
    @staticmethod
    def builder(model):
        model.fiber_section_length = 20          # µm (parallel fiber section length)
        model.fiber_segment_length = 7
        model.ascending_axon_length = 126       # µm
        model.parallel_fiber_length = 2000      # µm
        model.build_soma()
        model.build_dendrites()
        model.build_hillock()
        model.build_ascending_axon()
        model.build_parallel_fiber()

    morphologies = [builder]

    def build_soma(self):
        self.soma = [p.Section()]
        self.soma[0].set_dimensions(length=5.62232, diameter=5.8)
        self.soma[0].set_segments(1)
        self.soma[0].add_3d([self.position, self.position + [0., 5.62232, 0.]])

    def build_dendrites(self):
        self.dend = []
        for i in range(4):
            dendrite = p.Section()
            self.dend.append(dendrite)
            dendrite_position = self.position.copy()
            # Shift the dendrites a little bit for voxelization
            dendrite_position[0] += (i - 1.5) * 2
            dendrite.set_dimensions(length=15, diameter=0.75)
            points = []
            for j in range(10):
                pt = dendrite_position.copy()
                pt[1] -= dendrite.L * j / 10
                points.append(pt)
            dendrite.add_3d([[p[0], p[1], p[2]] for p in points])
            dendrite.connect(self.soma[0],0)

    def build_hillock(self):
        hillock = p.Section()
        hillock.set_dimensions(length=1,diameter=1.5)
        hillock.set_segments(1)
        hillock.add_3d([self.position + [0., 5.62232, 0.], self.position + [0., 6.62232, 0.]])
        hillock.labels = ["axon_hillock"]
        hillock.connect(self.soma[0], 0)

        ais = p.Section(name="axon_initial_segment")
        ais.labels = ["axon_initial_segment"]
        ais.set_dimensions(length=10,diameter=0.7)
        ais.set_segments(1)
        ais.add_3d([self.position + [0., 6.62232, 0.], self.position + [0., 16.62232, 0.]])
        ais.connect(hillock, 1)

        self.axon = [hillock, ais]
        self.axon_hillock = hillock
        self.axon_initial_segment = ais

    def build_ascending_axon(self):
        seg_length = self.fiber_segment_length
        n = int(self.ascending_axon_length / seg_length)

        self.ascending_axon = p.Section()
        self.ascending_axon.labels = ["ascending_axon"]
        self.ascending_axon.nseg = int(n)
        self.ascending_axon.L = self.ascending_axon_length
        self.ascending_axon.diam = 0.3
        previous_section = self.axon_initial_segment
        self.axon.append(self.ascending_axon)
        self.ascending_axon.connect(previous_section)

        y = 16.62232

        # Extract a set of intermediate points between start and end of ascending_axon to improve voxelization in scaffold
        fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        points = [
            self.position
            + [0., (y + f * self.ascending_axon_length), 0.]
            for f in fraction
        ]

        self.ascending_axon.add_3d(points)

        # Store the last used y position as the start for the parallel fiber
        self.y_pf = y + (seg_length * n)

    def build_parallel_fiber(self):
        section_length = self.fiber_section_length
        n = int(self.parallel_fiber_length / section_length)
        self.parallel_fiber = [p.Section(name='parellel_fiber_'+str(x)) for x in range(n)]
        # Use the last AA y as Y for the PF
        y = self.y_pf
        center = self.position[2]
        for id, section in enumerate(self.parallel_fiber):
            section.labels = ["parallel_fiber"]
            section.set_dimensions(length=section_length, diameter=0.3)
            sign = 1 - (id % 2) * 2
            z = floor(id / 2) * section_length
            section.add_3d([
                self.position + [0., y, center + sign * z],
                self.position + [0., y, center + sign * (z + section_length)]
            ])
            if id < 2:
                section.connect(self.ascending_axon)
            else:
                section.connect(self.parallel_fiber[id - 2])
            z += section_length
        self.axon.extend(self.parallel_fiber)


class GranuleCell(GranuleBase, DbbsNeuronModel):
    synapse_types = {
        "AMPA": {
            "point_process": ('AMPA', 'granule'),
            "attributes": {
                "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 1400, "U": 0.43
            }
        },
        "NMDA": {
            "point_process": ('NMDA', 'granule'),
            "attributes": {
                "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 23500, "U": 0.43
            }
        },
        "GABA": {
            "point_process": ('GABA', 'granule'),
            "attributes": {"U": 0.35}
        }
    }

    section_types = {
        "soma": {
            "mechanisms": ['Leak', 'Kv3_4', 'Kv4_3', 'Kir2_3', 'Ca', 'Kv1_1', 'Kv1_5', 'Kv2_2', ('cdp5', 'CR')],
            "attributes": {
                "Ra": 100, "cm": 2,
                ("e","Leak"): -60, "ek": -88, "eca": 137.5,
                ("gmax", "Leak"): 0.00029038073716,
                ("gkbar", "Kv3_4"): 0.00076192450951999995,
                ("gkbar", "Kv4_3"): 0.0028149683906099998,
                ("gkbar", "Kir2_3"): 0.00074725514701999996,
                ("gcabar", "Ca"): 0.00060938071783999998,
                ("gbar", "Kv1_1"):  0.0056973826455499997,
                ("gKur", "Kv1_5"):  0.00083407556713999999,
                ("gKv2_2bar", "Kv2_2"): 1.203410852e-05
            }
        },
        "dendrites": {
            "synapses": ['NMDA', 'AMPA', 'GABA'],
            "mechanisms": ['Leak', ('Leak', 'GABA'), 'Ca', 'Kca1_1', 'Kv1_1', ('cdp5', 'CR')],
            "attributes": {
                "Ra": 100, "cm": 2.5,
                ("e", "Leak"):  -60, "ek": -88, "eca": 137.5,
                ("gmax", "Leak"): 0.00025029700736999997,
                ("gcabar", "Ca"): 0.0050012800845900002,
                ("gbar", "Kca1_1"): 0.010018074546510001,
                ("gbar", "Kv1_1"): 0.00381819207934
            }
        },
        "axon": {
            "mechanisms": [], "attributes": {}
        },
        "ascending_axon": {
            "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
            "attributes": {
                "Ra": 100, "cm": 1,
                "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                ("gnabar", "Na"): 0.026301636815019999,
                ("gkbar", "Kv3_4"): 0.00237386061632,
                ("gmax", "Leak"):  9.3640921249999996e-05,
                ("gcabar", "Ca"): 0.00068197420273000001,
            }
        },
        "parallel_fiber": {
            "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
            "attributes": {
            "L": 20, "diam": 0.15, "Ra": 100, "cm": 1,
            "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
            ("gnabar", "Na"): 0.017718484492610001,
            ("gkbar", "Kv3_4"): 0.0081756804703699993,
            ("gmax", "Leak"): 3.5301616000000001e-07,
            ("gcabar", "Ca"): 0.00020856833529999999
            }
        },
        "axon_initial_segment": {
            "mechanisms": [('Na', 'granule_cell_FHF'), 'Kv3_4', 'Leak', 'Ca', 'Km', ('cdp5', 'CR')],
            "attributes": {
                "Ra": 100, "cm": 1,
                "ena": 87.39, "ek": -88, "eca": 137.5, ("e","Leak"):  -60,
                ("gnabar", "Na"): 1.28725006737226,
                ("gkbar", "Kv3_4"): 0.0064959534065400001,
                ("gmax", "Leak"): 0.00029276697557000002,
                ("gcabar", "Ca"):  0.00031198539471999999,
                ("gkbar", "Km"):  0.00056671971737000002
            }
        },
        "axon_hillock": {
            "mechanisms": ['Leak', ('Na', 'granule_cell_FHF'), 'Kv3_4', 'Ca', ('cdp5', 'CR')],
            "attributes": {
                "Ra": 100, "cm": 2,
                ("e","Leak"):  -60, "ena": 87.39, "ek": -88, "eca": 137.5,
                ("gmax", "Leak"): 0.00036958189720000001,
                ("gnabar", "Na"): 0.0092880585146199995,
                ("gkbar", "Kv3_4"): 0.020373463109149999,
                ("gcabar", "Ca"): 0.00057726155447
            }
        }
    }

class GranuleCellMildAdapting(GranuleBase, DbbsNeuronModel):
        synapse_types = {
            "AMPA": {
                "point_process": ('AMPA', 'granule'),
                "attributes": {
                    "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 1400, "U": 0.43
                }
            },
            "NMDA": {
                "point_process": ('NMDA', 'granule'),
                "attributes": {
                    "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 23500, "U": 0.43
                }
            },
            "GABA": {
                "point_process": ('GABA', 'granule'),
                "attributes": {"U": 0.35}
            }
        }

        section_types = {
            "soma": {
                "mechanisms": ['Leak', 'Kv3_4', 'Kv4_3', 'Kir2_3', 'Ca', 'Kv1_1', 'Kv1_5', 'Kv2_2', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2,
                    ("e","Leak"): -60, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00020821612897999999,
                    ("gkbar", "Kv3_4"): 0.00053837153610999998,
                    ("gkbar", "Kv4_3"): 0.0032501728450999999,
                    ("gkbar", "Kir2_3"): 0.00080747403035999997,
                    ("gcabar", "Ca"): 0.00066384354030999998,
                    ("gbar", "Kv1_1"):  0.0046520692281700003,
                    ("gKur", "Kv1_5"): 0.00106988075956,
                    ("gKv2_2bar", "Kv2_2"): 2.5949576899999998e-05
                }
            },
            "dendrites": {
                "synapses": ['NMDA', 'AMPA', 'GABA'],
                "mechanisms": ['Leak', ('Leak', 'GABA'), 'Ca', 'Kca1_1', 'Kv1_1', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2.5,
                    ("e", "Leak"):  -60, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00020424219215,
                    ("gcabar", "Ca"): 0.01841833779253,
                    ("gbar", "Kca1_1"): 0.02998872868395,
                    ("gbar", "Kv1_1"): 0.00010675447184
                }
            },
            "axon": {
                "mechanisms": [], "attributes": {}
            },
            "ascending_axon": {
                "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 1,
                    "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                    ("gnabar", "Na"): 0.029973709719629999,
                    ("gkbar", "Kv3_4"): 0.0046029972380800003,
                    ("gmax", "Leak"):  7.8963697590000003e-05,
                    ("gcabar", "Ca"): 0.00059214434259999998,
                }
            },
            "parallel_fiber": {
                "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                "L": 20, "diam": 0.15, "Ra": 100, "cm": 1,
                "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                ("gnabar", "Na"): 0.01896618618573,
                ("gkbar", "Kv3_4"): 0.0094015060525799998,
                ("gmax", "Leak"): 4.1272473000000001e-07,
                ("gcabar", "Ca"): 0.00064742320254000001
                }
            },
            "axon_initial_segment": {
                "mechanisms": [('Na', 'granule_cell_FHF'), 'Kv3_4', 'Leak', 'Ca', 'Km', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 1,
                    "ena": 87.39, "ek": -88, "eca": 137.5, ("e","Leak"):  -60,
                    ("gnabar", "Na"): 1.06883116205825,
                    ("gkbar", "Kv3_4"): 0.034592458064240002,
                    ("gmax", "Leak"): 0.00025011065810000001,
                    ("gcabar", "Ca"):  0.00011630629281,
                    ("gkbar", "Km"):  0.00044764153078999998
                }
            },
            "axon_hillock": {
                "mechanisms": ['Leak', ('Na', 'granule_cell_FHF'), 'Kv3_4', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2,
                    ("e","Leak"):  -60, "ena": 87.39, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00025295417368000002,
                    ("gnabar", "Na"): 0.011082499796400001,
                    ("gkbar", "Kv3_4"): 0.050732563882920002,
                    ("gcabar", "Ca"): 0.00028797253573000002
                }
            }
        }

class GranuleCellAdapting(GranuleBase, DbbsNeuronModel):
        synapse_types = {
            "AMPA": {
                "point_process": ('AMPA', 'granule'),
                "attributes": {
                    "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 1400, "U": 0.43
                }
            },
            "NMDA": {
                "point_process": ('NMDA', 'granule'),
                "attributes": {
                    "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 23500, "U": 0.43
                }
            },
            "GABA": {
                "point_process": ('GABA', 'granule'),
                "attributes": {"U": 0.35}
            }
        }

        section_types = {
            "soma": {
                "mechanisms": ['Leak', 'Kv3_4', 'Kv4_3', 'Kir2_3', 'Ca', 'Kv1_1', 'Kv1_5', 'Kv2_2', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2,
                    ("e","Leak"): -60, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00027672909671000001,
                    ("gkbar", "Kv3_4"): 0.00373151328841,
                    ("gkbar", "Kv4_3"): 0.0027313162972600002,
                    ("gkbar", "Kir2_3"): 0.00094360184424999995,
                    ("gcabar", "Ca"): 0.00029165028328999998,
                    ("gbar", "Kv1_1"):  0.0031675812802999998,
                    ("gKur", "Kv1_5"): 0.00107176612352,
                    ("gKv2_2bar", "Kv2_2"): 6.710092624e-05
                }
            },
            "dendrites": {
                "synapses": ['NMDA', 'AMPA', 'GABA'],
                "mechanisms": ['Leak', ('Leak', 'GABA'), 'Ca', 'Kca1_1', 'Kv1_1', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2.5,
                    ("e", "Leak"):  -60, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00029871180381000001,
                    ("gcabar", "Ca"): 0.024687091736070001,
                    ("gbar", "Kca1_1"): 0.01185742892862,
                    ("gbar", "Kv1_1"): 0.00015853886699000001
                }
            },
            "axon": {
                "mechanisms": [], "attributes": {}
            },
            "ascending_axon": {
                "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 1,
                    "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                    ("gnabar", "Na"): 0.025441894508310001,
                    ("gkbar", "Kv3_4"): 0.0046504514953399998,
                    ("gmax", "Leak"):  5.3037170669999997e-05,
                    ("gcabar", "Ca"): 0.00031374692347000001,
                }
            },
            "parallel_fiber": {
                "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                "L": 20, "diam": 0.15, "Ra": 100, "cm": 1,
                "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                ("gnabar", "Na"): 0.0142518259615,
                ("gkbar", "Kv3_4"): 0.0098649550733799999,
                ("gmax", "Leak"): 1.4118927999999999e-07,
                ("gcabar", "Ca"): 0.00024821458382999999
                }
            },
            "axon_initial_segment": {
                "mechanisms": [('Na', 'granule_cell_FHF'), 'Kv3_4', 'Leak', 'Ca', 'Km', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 1,
                    "ena": 87.39, "ek": -88, "eca": 137.5, ("e","Leak"):  -60,
                    ("gnabar", "Na"): 1.5810107836409499,
                    ("gkbar", "Kv3_4"): 0.039582385081389997,
                    ("gmax", "Leak"): 0.00025512657995000002,
                    ("gcabar", "Ca"): 0.00038160760886000002,
                    ("gkbar", "Km"):  0.00049717923887
                }
            },
            "axon_hillock": {
                "mechanisms": ['Leak', ('Na', 'granule_cell_FHF'), 'Kv3_4', 'Ca', ('cdp5', 'CR')],
                "attributes": {
                    "Ra": 100, "cm": 2,
                    ("e","Leak"):  -60, "ena": 87.39, "ek": -88, "eca": 137.5,
                    ("gmax", "Leak"): 0.00031475453130000002,
                    ("gnabar", "Na"): 0.020910983616370001,
                    ("gkbar", "Kv3_4"): 0.03097630887484,
                    ("gcabar", "Ca"): 0.00019803691988000001
                }
            }
        }

class GranuleCellAccelerate(GranuleBase, DbbsNeuronModel):
                synapse_types = {
                    "AMPA": {
                        "point_process": ('AMPA', 'granule'),
                        "attributes": {
                            "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 1400, "U": 0.43
                        }
                    },
                    "NMDA": {
                        "point_process": ('NMDA', 'granule'),
                        "attributes": {
                            "tau_facil": 5, "tau_rec": 8, "tau_1": 1, "gmax": 23500, "U": 0.43
                        }
                    },
                    "GABA": {
                        "point_process": ('GABA', 'granule'),
                        "attributes": {"U": 0.35}
                    }
                }

                section_types = {
                    "soma": {
                        "mechanisms": ['Leak', 'Kv3_4', 'Kv4_3', 'Kir2_3', 'Ca', 'Kv1_1', 'Kv1_5', 'Kv2_2', ('cdp5', 'CAM')],
                        "attributes": {
                            "Ra": 100, "cm": 2,
                            ("e","Leak"): -60, "ek": -88, "eca": 137.5,
                            ("gmax", "Leak"): 0.00029038073716,
                            ("gkbar", "Kv3_4"): 0.00076192450951999995,
                            ("gkbar", "Kv4_3"): 0.0028149683906099998,
                            ("gkbar", "Kir2_3"): 0.00074725514701999996,
                            ("gcabar", "Ca"): 0.00060938071783999998,
                            ("gbar", "Kv1_1"):  0.0056973826455499997,
                            ("gKur", "Kv1_5"): 0.00083407556713999999,
                            ("gKv2_2bar", "Kv2_2"): 1.203410852e-05
                        }
                    },
                    "dendrites": {
                        "synapses": ['NMDA', 'AMPA', 'GABA'],
                        "mechanisms": ['Leak', ('Leak', 'GABA'), 'Ca', 'Kca1_1', 'Kv1_1', ('cdp5', 'CAM')],
                        "attributes": {
                            "Ra": 100, "cm": 2.5,
                            ("e", "Leak"):  -60, "ek": -88, "eca": 137.5,
                            ("gmax", "Leak"): 0.00025029700736999997,
                            ("gcabar", "Ca"): 0.0050012800845900002,
                            ("gbar", "Kca1_1"): 0.010018074546510001,
                            ("gbar", "Kv1_1"): 0.00381819207934
                        }
                    },
                    "axon": {
                        "mechanisms": [], "attributes": {}
                    },
                    "ascending_axon": {
                        "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CAM')],
                        "attributes": {
                            "Ra": 100, "cm": 1,
                            "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                            ("gnabar", "Na"): 0.026301636815019999,
                            ("gkbar", "Kv3_4"): 0.00237386061632,
                            ("gmax", "Leak"):  9.3640921249999996e-05,
                            ("gcabar", "Ca"): 0.00068197420273000001,
                        }
                    },
                    "parallel_fiber": {
                        "mechanisms": [('Na', 'granule_cell'), 'Kv3_4', 'Leak', 'Ca', ('cdp5', 'CAM')],
                        "attributes": {
                        "L": 20, "diam": 0.15, "Ra": 100, "cm": 1,
                        "ena": 87.39, "ek": -88, ("e","Leak"):  -60, "eca": 137.5,
                        ("gnabar", "Na"): 0.017718484492610001,
                        ("gkbar", "Kv3_4"): 0.0081756804703699993,
                        ("gmax", "Leak"): 3.5301616000000001e-07,
                        ("gcabar", "Ca"): 0.00020856833529999999
                        }
                    },
                    "axon_initial_segment": {
                        "mechanisms": [('Na', 'granule_cell_FHF'), 'Kv3_4', 'Leak', 'Ca', 'Km', ('cdp5', 'CAM')],
                        "attributes": {
                            "Ra": 100, "cm": 1,
                            "ena": 87.39, "ek": -88, "eca": 137.5, ("e","Leak"):  -60,
                            ("gnabar", "Na"): 1.28725006737226,
                            ("gkbar", "Kv3_4"): 0.0064959534065400001,
                            ("gmax", "Leak"): 0.00029276697557000002,
                            ("gcabar", "Ca"): 0.00031198539471999999,
                            ("gkbar", "Km"):  0.00056671971737000002
                        }
                    },
                    "axon_hillock": {
                        "mechanisms": ['Leak', ('Na', 'granule_cell_FHF'), 'Kv3_4', 'Ca', ('cdp5', 'CAM')],
                        "attributes": {
                            "Ra": 100, "cm": 2,
                            ("e","Leak"):  -60, "ena": 87.39, "ek": -88, "eca": 137.5,
                            ("gmax", "Leak"): 0.00036958189720000001,
                            ("gnabar", "Na"): 0.0092880585146199995,
                            ("gkbar", "Kv3_4"): 0.020373463109149999,
                            ("gcabar", "Ca"): 0.00057726155447
                        }
                    }
                }
