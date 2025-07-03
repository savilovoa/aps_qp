import plotly.graph_objects as go

fig = go.Figure(go.Table(
        header=dict(
            values=[""] + list("abcd"),
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[["red","white","blue"],[5,3,2],[8,4,5],[9,5,3],[5,2,1]],
            align = "left")
    ))
f_name = "res.html"
with open(f_name, "w", encoding="utf8") as f:
    f.write(fig.to_html())
fig.to_html("res.html")
