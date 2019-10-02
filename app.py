from waitress import serve

from flask import Flask, render_template, request
from python.functions import  get_pos_neg_words_df, full_shap_eval

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
    words = get_pos_neg_words_df(url)
    table = words.to_html()
    all_f = full_shap_eval(url)
    #f1_score = f1_ind(url)
    #text = force_text_ind(url)
    #force_plot = force_plot_ind(url)

    #return render_template("results.html", url=url, path='/static/images/shap_plot.png')
    return render_template('results.html', table=table, all_f=all_f)

#@app.route('/test')
#def show_plot():
#  lnprice=np.log(price)
#  plt.plot(lnprice)
#  plt.savefig('/static/images/shap_plot.png')
# return render_template('results.html', name = 'new_plot', url ='/static/images/shap_plot.png

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)

