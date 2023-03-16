import pandas as pd


adj_matrices = {'1962_2006_walmart_store_openings': pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),

                '2011_february_aa_flight_paths':   pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 1],
                                                                [0, 0, 0, 0, 0, 0, 0, 1],
                                                                [0, 0, 0, 0, 0, 1, 1, 0]]),

                'beers': pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),



                'horoscope_data': pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 1, 1, 1, 0, 1, 0]]),

                'laucnty16without1column':pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1, 1, 1, 1],
                                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0]]),

       'medicare_cost_without1': pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),






                'rapeandchildsexualabusedataPJ': pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]),


                'salaries-ai-jobs-net':  pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),


                'us-cities-top-1k-multi-year': pd.DataFrame([[0, 0, 0, 0, 0, 0],
                                                             [0, 0, 1, 0, 0, 0],
                                                             [0, 1, 0, 0, 0, 1],
                                                             [0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0],
                                                             [0, 0, 1, 0, 0, 0]]),


                'US-shooting-incidents': pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

                'volcano_db': pd.DataFrame([  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),


                'data1':  pd.DataFrame([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

                'Tips_data': pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 1, 1, 1, 1],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0]]),

                'solar':  pd.DataFrame([[0, 1, 1, 0, 1],
                                        [1, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0]]),

                'school_earnings':pd.DataFrame([[0, 1, 1, 1],
                                                [1, 0, 0, 0],
                                                [1, 0, 0, 0],
                                                [1, 0, 0, 0]]),


                'sales_success': pd.DataFrame([ [0, 0, 0, 1, 1],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0]]),

                'motor_trend_car_road_tests':pd.DataFrame([ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),

                'job_automation_probability' : pd.DataFrame([   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),



                 'iris': pd.DataFrame([ [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 1],
                                        [1, 1, 1, 1, 0]]),    

                'gapminder_unfiltered':   pd.DataFrame([[0, 0, 0, 0, 0, 0],
                                                        [0, 0, 0, 1, 1, 1],
                                                        [0, 0, 0, 1, 1, 1],
                                                        [0, 1, 1, 0, 0, 0],
                                                        [0, 1, 1, 0, 0, 0],
                                                        [0, 1, 1, 0, 0, 0]]),

                'english_french': pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),

                'Emissions_Data': pd.DataFrame([[0, 0, 0, 1],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 1],
                                                [1, 0, 1, 0]]),

                'auto_mpg': pd.DataFrame([  [0, 1, 0, 0, 0, 0, 1],
                                            [1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [1, 0, 0, 1, 1, 0, 0]]),


                '2011_february_us_airport_traffic':   pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                                                    [0, 0, 1, 0, 0, 0, 0, 1],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 1, 0, 0, 0, 0]]),


                'diabetes':   pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 1, 0, 0, 1, 0, 0, 0]]),

                'minoritymajority':   pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

                'Antibiotics':  pd.DataFrame(  [[0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0]])


                }

