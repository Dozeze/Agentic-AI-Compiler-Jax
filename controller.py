# Controller

# PYTHON FILE ONLY! SMALL STEPS!


class Controller:
    def __init__(self, agent : str, prompt : str):
        """ What goes in -> agent (string), a version"""
        self.agent = agent
        self.history = []
        self.iteration = 0
        self.init_prompt = prompt

    def initHist(self, original_code):
        """ Initializes the hist for agent """
        #1. Compile code
        runtime = self.current_runtime(original_code)

        #2. Update hist
        self.history.append(
            {
                "iteration" : self.iteration,
                "code" : "Default",             #TODO
                "runtime" : runtime,            #TODO
                "changes" : "No changes"
                                                #Might give agent able to communicate to older versions for MEMORY / IMPROVEMENTS
            }
        )

    def updateHist(self, new_code):
        """ 
            Updates history AFTER an iteration is completed.    
        """

        runtime = self.current_runtime(new_code)

        old_code = self.history[-1]["code"]

        changes = self.changesPython(old_code, new_code)

        #will also include HLO / CUDA kernels in future #TODO
        self.history.append(
                {
                    "iteration" : self.iteration,
                    "code" : new_code,
                    "runtime" : runtime,
                    "changes" : changes
                }
        )

    # Kanske går in i agenten
    def provideTools(self):
        TOOL_SCHEMAS = [{
                        "type" : "function", 
                        "function" : 
                        {
                         "name" : "Name of func",
                         "description" : "Description of func",
                         "parameters" : "Parameters of func",
                         "properties" : 
                            {
                                "path" : {"type" : "string", "description" : "Path of the file to read"}
                            },
                        "required" : ["path"] 
                        }
                        }
                        ,
                        "Next one ..."
                        ]

    def current_runtime(self, new_code):
        # TODO !
        return -1

    def changesPython(self, old_code, new_code):
        if (self.iteration == 0):
            return ""
        else:
            return "Not updated yet" #TODO

    def provide_hist(self):
        return self.history

    def provide_prompt(self):
        return self.init_prompt
    


    
    