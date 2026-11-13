/* Delegated events survive Dash's live table updates; previews need no round trip. */
(function () {
    let active = null;
    let popup = null;
    function hide() {
        if (active) active.removeAttribute('aria-describedby');
        if (popup) popup.remove();
        popup = null;
        active = null;
    }
    function show(button) {
        if (active === button) return;
        hide();
        let data;
        try { data = JSON.parse(button.dataset.loadPreview); } catch (_) { return; }
        active = button;
        popup = document.createElement('div');
        popup.id = 'load-preview-popover';
        popup.className = 'load-preview-popover';
        popup.setAttribute('role', 'tooltip');
        const title = document.createElement('strong');
        title.textContent = data.title;
        const weight = document.createElement('div');
        weight.textContent = data.weight + ' · logarithmic bins';
        const image = document.createElement('img');
        image.src = data.image;
        image.alt = data.kind === 'percentile' ? 'Signed distribution with percentile threshold and shaded outer tail' : 'Signed distribution with zero and the hovered extreme marked';
        const summary = document.createElement('div');
        summary.className = 'preview-statistics';
        summary.textContent = data.summary;
        const scope = document.createElement('small');
        scope.textContent = data.scope;
        const hint = document.createElement('small');
        hint.textContent = 'Click to open this component in the larger histogram.';
        popup.append(title, weight, image, summary, scope, hint);
        document.body.appendChild(popup);
        button.setAttribute('aria-describedby', popup.id);
        const bounds = button.getBoundingClientRect();
        const width = popup.offsetWidth;
        const height = popup.offsetHeight;
        const left = Math.min(Math.max(8, bounds.left + bounds.width / 2 - width / 2), Math.max(8, window.innerWidth - width - 8));
        const top = bounds.bottom + 8 + height <= window.innerHeight ? bounds.bottom + 8 : Math.max(8, bounds.top - height - 8);
        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
    }
    document.addEventListener('pointerover', function (event) {
        const button = event.target.closest('.load-value');
        if (button) show(button);
    });
    document.addEventListener('pointerout', function (event) {
        if (active && active.contains(event.target) && !active.contains(event.relatedTarget)) hide();
    });
    document.addEventListener('focusin', function (event) {
        const button = event.target.closest('.load-value');
        if (button) show(button);
    });
    document.addEventListener('focusout', function (event) {
        if (active && active.contains(event.target)) hide();
    });
    document.addEventListener('keydown', function (event) {
        if (event.key !== 'Escape') return;
        hide();
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props('bin-filter', {data: null});
        }
    });
    window.addEventListener('scroll', hide, {passive: true});
    window.addEventListener('resize', hide);
    document.addEventListener('click', function (event) {
        const button = event.target.closest('.load-value');
        if (!button || !window.dash_clientside || !window.dash_clientside.set_props) return;
        const data = JSON.parse(button.dataset.loadPreview);
        hide();
        const set = window.dash_clientside.set_props;
        set('interface', {value: data.interface});
        set('metric', {value: data.metric});
        set('extreme', {value: data.side});
        set('bin-filter', {data: null});
        set('histogram-focus', {data: {path: data.path, part: data.part, interface: data.interface,
            metric: data.metric, side: data.side, value: data.value, kind: data.kind, q: data.q}});
        const section = document.getElementById('histogram-section');
        if (section) section.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start'});
    });
})();
