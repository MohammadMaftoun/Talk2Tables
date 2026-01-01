from agents_module import PlannerAgent, CodeGeneratorAgent, VerifierAgent, ExecutionAgent, ExplainerAgent
from utils_module import get_data_info
import pandas as pd

# Create sample dataframe
df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [2, 3, 4, 5, 6],
    'category': ['x', 'y', 'x', 'y', 'x']
})

# Get data info
info = get_data_info(df)

# Initialize agents
planner = PlannerAgent()
codegen = CodeGeneratorAgent()
verifier = VerifierAgent()
executor = ExecutionAgent()
explainer = ExplainerAgent()

query = 'Describe my dataset'
plan = planner.plan(query, info)
print('Plan: ', plan['plan_type'])

code_bundle = codegen.generate(plan, info)
print('Generated code length:', len(code_bundle['code'].split('\n')))

verification = verifier.verify(code_bundle['code'])
print('Verification safe:', verification['safe'])
if not verification['safe']:
    print('Issues:', verification['issues'])

execution = executor.execute(code_bundle['code'], df)
print('Execution success:', execution['success'])
print('Execution output:\n', execution['output'])

explanation = explainer.explain(execution, plan)
print('\nExplanation:\n', explanation['explanation'])