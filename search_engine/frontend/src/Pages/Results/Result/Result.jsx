const utf8 = require('utf8');
function Result({ result }) {

    // const { title, description, url } = result;
    const { title, url } = result;

    return (
        <div className="result">
            <a href={url} className="result__url"><h2 className="result__title">{decodeURIComponent(title)}</h2></a>
            {/* <p className="result__description">{description}</p> */}
        </div>
    )
}

export default Result;