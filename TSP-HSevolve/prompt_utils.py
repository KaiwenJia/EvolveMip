#***************************************************************************
#prompt utils: including all prompts using in the projects
#***************************************************************************
# refine the prompt template, add instructs
OR_prompt_dict = {"TSP":
                  f"""
    As a senior combinatorial optimization expert specialized in heuristics strategy for TSP, 
    you are asked to improve the following code blocks in a TSP Python code file. Your optimizations should focus on enhancing solution quality, accelerating convergence, and avoiding local optima while maintaining compatibility with the original code structure.

    Here is the code block for initial solution generation:
    <insert_initial_solution>
    - Key improvement directions for initial solutions:
      1. Propose adaptive starting point selection strategies (e.g., based on coordinate distribution, distance centrality, or problem-specific features)
      2. Enhance greedy methods with heuristic adjustments (e.g., weighted nearest neighbor, multi-start initialization)
      3. Integrate randomized elements to increase solution diversity without excessive computational overhead

    Here is the main algorithm logic for searching the best solution:
    <insert_main_algorithm>
    - Key improvement directions for the main algorithm:
      1. Optimize local search operators (e.g., 2-opt, 3-opt) with adaptive step sizes or neighborhood restriction
      2. Refine perturbation strategies (timing, intensity) to balance exploration and exploitation
      3. Improve heuristic weight adjustment in guided local search (dynamic thresholds based on distance matrix statistics)
      4. Add termination condition enhancements (adaptive iteration limits based on solution improvement rate)

    Your response must include:
    1. Updated code blocks with clear comments explaining changes
    2. Rationale for each modification (linking to TSP optimization principles)
    3. Maintenance of original input/output data structures (e.g., np.array for tours, distance matrix format)
    4. Compatibility with existing helper functions (e.g., tour_cost_2End, guided_local_search parameters)

    Prioritize practical improvements over theoretical complexity, ensuring the modified code runs efficiently for TSP instances with 50-200 nodes.
        """}

prompt_step1_dict = {"conflict_check":
    f"""
    As a senior optimization expert and seasoned industrial engineer, you are asked to assess the fit between a specific optimization model and a real-world industry scenario.

    the given industry scenario list:
    <insert_scenario>

    ## Subclass of Optimization Modeling Problems:
    <insert_basic_model>
    
    Your response should follow two steps:
    1 <think>
    the first step rovide a detailed reasoning. Describe the key characteristics of the industry scenario that align with the core principles and constraints of the optimization model. If it is not a good fit, explain the mismatches between the problem and the model.
     </thinkchenyitian
     >

    2,<judge>
    State whether the specified optimization problem subclass is applicable to the given industry scenario.Output(only Part 2) strictly in the following format (MUST be uppercase, appear alone, and exactly as shown):
    Judgment: "YES" or "NO"
     <\judge>
    The output must be in Markdown format, with each step enclosed in the specified tags.
    """
        }

gurobi_q2mc_roles={
        "role1":"A highly skilled Python engineer and optimization specialist with deep expertise in operations research and the gurobi solver.",
        "role2":"An optimization expert and  Python engineer  specializing in operations research and the gurobi solver.",
        "role3":"A Python engineer and optimization specialist with a strong background in operations research and the gurobi solver.",
        "role4":"A skilled Python engineer and optimization specialist proficient in operations research and the gurobi solver.",
        "role5":"A results-driven Python engineer and optimization expert with a strong foundation in operations research and the gurobi solver.",
        "role6": "A seasoned operations research scientist and Python developer, leveraging advanced optimization techniques and the Gurobi solver to tackle complex business challenges.",
        "role7": "An innovative optimization modeler and Python programmer, specializing in the development and implementation of high-performance solutions using operations research methodologies and the Gurobi optimization suite.",
        "role8": "A pragmatic problem-solver with expertise in operations research, proficient in Python and the Gurobi API, focused on translating real-world scenarios into efficient and scalable optimization models.",
        "role9": "A meticulous optimization analyst and Python coder, deeply familiar with the theoretical underpinnings of operations research and the practical application of the Gurobi solver for achieving optimal outcomes.",
        "role10": "A strategic optimization architect and Python implementation specialist, with a proven track record of designing and deploying robust operations research solutions powered by the Gurobi optimization engine."
    }

