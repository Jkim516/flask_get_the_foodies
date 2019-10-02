from waitress import serve

from flask import Flask, render_template, request

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")

@app.route("/get_results", methods=["POST"])
def get_results():
    data = request.form
    print(data)
    # user_number = int(data["number"])
    # user_number_doubled = user_number * 2

    url = data["yelp_url"] 

    return render_template("results.html", url=url, path='/static/images/shap_plot.png')

#@app.route('/test')
#def show_plot():
#  lnprice=np.log(price)
#  plt.plot(lnprice)
#  plt.savefig('/static/images/shap_plot.png')
# return render_template('results.html', name = 'new_plot', url ='/static/images/shap_plot.png

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)

