from riskpy import UnderwritingApp, FactorModel

print("Loading Actuarial Pricing Suite with Advanced Visualizations...")

app = UnderwritingApp(title="Enterprise Actuarial Suite", excel_template="corporate_layout.xlsx")

app.add_field("state", "Location State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Client Age", "B")

model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "CA", 2.0)
app.set_factor_model(model)

app.run()
