import { useState } from 'react';
import { FaSearch } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

function Search() {

    const navigate = useNavigate();

    const [searchText, setSearchText] = useState('');

    const submitSearch = (e) => {
        e.preventDefault();

        if (!searchText) return;
        navigate(`/${searchText}`)
    }

    return (
        <div className="search">
            <h1 className="search__title">Search</h1>
            <form className="search-bar" onSubmit={(e) => submitSearch(e)}>
                < FaSearch className='search-bar__icon' />
                <input className="search-bar__input" value={searchText} onChange={(e) => setSearchText(e.target.value)} />
            </form>
            <button className='search__submit-btn' onClick={submitSearch}>Search</button>
        </div>
    );
}

export default Search;
