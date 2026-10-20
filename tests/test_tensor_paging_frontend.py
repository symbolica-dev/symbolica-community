"""Exercise Symbolica's actual shared widget with the native tensor provider."""

import os
import re

import pytest
from symbolica import E, MathMLViewer, data
from symbolica.community.tensor import Representation, TensorName

pw = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def browser_page():
    with pw.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(
                **(
                    {"executable_path": os.environ["CHROMIUM_EXECUTABLE"]}
                    if "CHROMIUM_EXECUTABLE" in os.environ
                    else {}
                )
            )
        except pw.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        try:
            yield browser.new_page(viewport={"width": 1600, "height": 1000})
        finally:
            browser.close()


@pytest.fixture(params=("chromium", "webkit"))
def mathml_browser_page(request):
    """Exercise native MathML geometry in both independent browser engines."""
    with pw.sync_playwright() as playwright:
        engine = request.param
        executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
        try:
            browser = getattr(playwright, engine).launch(
                **({"executable_path": executable} if executable else {})
            )
        except pw.Error as error:
            pytest.skip(f"{engine} unavailable: {error}")
        try:
            yield browser.new_page(viewport={"width": 1600, "height": 1000})
        finally:
            browser.close()


def mount_viewer(page, viewer):
    widget = viewer._get_widget()
    # Use the exact verified resource shipped with the extension. Network or
    # system-font fallback can otherwise hide geometry and glyph regressions.
    font = data.read("render/fonts/STIXTwoMath-Regular.woff2")
    page.route(
        "**/STIXTwoMath-Regular.woff2",
        lambda route: route.fulfill(
            body=font,
            content_type="font/woff2",
            headers={"Access-Control-Allow-Origin": "*"},
        ),
    )
    page.set_content('<div id="viewer"></div>')
    page.evaluate(
        """async ({esm, page}) => {
        const module = await import(URL.createObjectURL(new Blob([esm], {type:'text/javascript'})));
        // Live notebook widgets render inside a shadow root, where :root
        // variables from the tensor stylesheet are unavailable.
        const root = document.querySelector('#viewer').attachShadow({mode:'open'});
        const element = document.createElement('div');
        root.appendChild(element);
        const fields = {page, preview:{}}, callbacks = new Map();
        window.messages = [];
        window.setField = ([name, value]) => {
            fields[name] = value;
            for (const callback of callbacks.get(`change:${name}`) || []) callback();
        };
        window.callbackCount = () => [...callbacks.values()].reduce((n,s)=>n+s.size,0);
        window.dispose = module.default.render({el:element, model:{
            get:name=>fields[name], send:message=>messages.push(message),
            on:(name, callback)=>{
                if (!callbacks.has(name)) callbacks.set(name,new Set());
                callbacks.get(name).add(callback);
            },
            off:(name, callback)=>callbacks.get(name)?.delete(callback)
        }});
    }""",
        {"esm": widget._esm, "page": viewer._page()},
    )
    loaded = page.evaluate(
        """async () => {
        const faces = await Promise.race([
            (async () => {
                const loaded = await document.fonts.load('21px "STIX Two Math"', '𝑖𝑘μν');
                await document.fonts.ready;
                return loaded;
            })(),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Math font did not load')), 10000))
        ]);
        return faces.length > 0 && document.fonts.check('21px "STIX Two Math"', '𝑖𝑘μν');
    }"""
    )
    assert loaded
    font_faces = page.evaluate(
        """() => [...document.fonts]
            .filter(face => face.family.replaceAll('"', '') === 'STIX Two Math')
            .map(face => face.status)"""
    )
    # The upstream widget registers the font once in the document. Tensor page
    # styles must not create extra shadow-root faces when the layout changes.
    assert font_faces == ["loaded"]
    return widget, page.locator("#viewer")


def mathml_geometry(main):
    """Measure fraction children, including table children that WebKit mislays."""
    return main.evaluate(
        """element => {
        const box = node => {
            const r = node.getBoundingClientRect();
            return {left:r.left, right:r.right, top:r.top, bottom:r.bottom,
                width:r.width, height:r.height, centerX:(r.left+r.right)/2};
        };
        const visible = node => {
            for (let p=node; p && p!==element; p=p.parentElement) {
                if (p.localName === 'mphantom' || getComputedStyle(p).display === 'none') return false;
            }
            return node.getClientRects().length > 0;
        };
        const math = element.querySelector(':scope > math');
        return {
            math:box(math),
            fontSize:Number.parseFloat(getComputedStyle(math).fontSize),
            availableWidth:element.parentElement.clientWidth,
            scrollWidth:element.scrollWidth,
            fractions:[...math.querySelectorAll('mfrac')]
                .filter(node => visible(node) && node.getAttribute('bevelled') !== 'true')
                .map(node => ({fraction:box(node),
                    numerator:box(node.children[0]), denominator:box(node.children[1]),
                    children:[...node.children].map(child => child.localName)}))
        };
    }"""
    )


def assert_stacked_fractions(geometry):
    assert geometry["fractions"], geometry
    for fraction in geometry["fractions"]:
        numerator, denominator = fraction["numerator"], fraction["denominator"]
        assert numerator["bottom"] <= denominator["top"] + 2, fraction
        assert abs(numerator["centerX"] - denominator["centerX"]) <= 2, fraction
        assert fraction["fraction"]["width"] + 2 >= max(
            numerator["width"], denominator["width"]
        ), fraction


def settle_layout(page):
    page.evaluate(
        """async () => {
        await document.fonts.ready;
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
    }"""
    )


def product_geometry(main):
    """Measure the largest product, preserving each semantic factor's identity."""
    return main.evaluate(
        """element => {
        const products = [...element.querySelectorAll('[data-symbolica-product]')];
        const candidates = products.map(product => {
            const factors = [...product.querySelectorAll('[data-symbolica-factor]')]
                .filter(factor => factor.closest('[data-symbolica-product]') === product);
            const rows = [...product.children];
            const factorRows = new Set(factors.map(factor => {
                let row = factor;
                while (row.parentElement !== product) row = row.parentElement;
                return row;
            }));
            return {rows:factorRows.size, totalRows:rows.length,
                factors:factors.map(factor => ({
                html:factor.outerHTML,
                text:factor.textContent,
                width:factor.getBoundingClientRect().width,
                paths:[...factor.querySelectorAll('[data-symbolica-path]')]
                    .map(node => node.getAttribute('data-symbolica-path'))
            })), width:product.getBoundingClientRect().width,
                rowWidths:rows.map(row => row.getBoundingClientRect().width),
                availableWidth:element.parentElement.clientWidth,
                // Include the actual multiplication glyphs, not a character estimate.
                separatorWidths:[...product.querySelectorAll('mo')]
                    .filter(node => node.closest('[data-symbolica-product]') === product
                        && !node.closest('[data-symbolica-factor]'))
                    .map(node => node.getBoundingClientRect().width)};
        });
        return candidates.sort((a,b) => b.factors.length-a.factors.length)[0];
    }"""
    )


@pytest.mark.parametrize("kind", ("tensor", "expression"))
def test_product_rows_follow_rendered_width_and_zoom_preserves_layout(
    mathml_browser_page, kind
):
    rep = Representation.mink(4)
    value = TensorName("product_fit::T0")(rep("mu"))
    for index in range(1, 10):
        value *= TensorName(f"product_fit::T{index}")(rep(f"index{index}"))
    value /= 8
    original, structure = value.to_expression(), value.structure
    # Both providers use the same renderer, while only the tensor provider
    # replaces the long symbolic leaf with compact tensor notation.
    viewer = value.paged() if kind == "tensor" else original.mathml_viewer()
    page = mathml_browser_page
    widget, host = mount_viewer(page, viewer)
    viewer._on_message(widget, page.evaluate("messages[0]"), [])
    main = host.locator(".math-content")
    try:
        measurements = {}
        for width in (1400, 350, 1400):
            host.evaluate(
                "(element, width) => element.style.width = `${width}px`", width
            )
            settle_layout(page)
            geometry = mathml_geometry(main)
            assert_stacked_fractions(geometry)
            product = product_geometry(main)
            assert len(product["factors"]) == 10, product
            assert geometry["scrollWidth"] <= geometry["availableWidth"] + 2, geometry
            if width in measurements:
                assert product == measurements[width]
            measurements[width] = product
        wide, narrow = measurements[1400], measurements[350]
        # These final emitted factors demonstrably fit in the wide host.
        # Their row packing must change when the available width changes.
        measured_width = sum(factor["width"] for factor in wide["factors"]) + sum(
            wide["separatorWidths"]
        )
        assert measured_width < wide["availableWidth"] - 40, wide
        assert wide["rows"] == 1, wide
        assert narrow["rows"] > wide["rows"], (wide, narrow)
        assert [(factor["html"], factor["paths"]) for factor in wide["factors"]] == [
            (factor["html"], factor["paths"]) for factor in narrow["factors"]
        ]
        assert wide["width"] > narrow["width"] + 20, (wide, narrow)

        # Upstream zoom deliberately magnifies the chosen layout without
        # regrouping it. Overflow must remain reachable and reset reversible.
        host.evaluate(
            "(element, width) => element.style.width = `${width}px`",
            measured_width + 80,
        )
        settle_layout(page)
        unzoomed = product_geometry(main)
        assert unzoomed["rows"] == 1, unzoomed
        message_count = page.evaluate("messages.length")
        region = host.get_by_role("region", name="Expression", exact=True)
        region.focus()
        for _ in range(2):
            region.press("=")
        settle_layout(page)
        zoomed = product_geometry(main)
        assert zoomed["rows"] == unzoomed["rows"], (unzoomed, zoomed)
        assert [factor["text"] for factor in zoomed["factors"]] == [
            factor["text"] for factor in unzoomed["factors"]
        ]
        assert page.evaluate("messages.length") == message_count
        assert_stacked_fractions(mathml_geometry(main))
        scrolling = main.evaluate(
            """element => {
            const container = element.parentElement;
            container.scrollLeft = container.scrollWidth;
            const result = {left:container.scrollLeft,
                overflowing:container.scrollWidth > container.clientWidth};
            container.scrollLeft = 0;
            return result;
        }"""
        )
        assert scrolling["overflowing"] and scrolling["left"] > 0, scrolling
        region.press("0")
        settle_layout(page)
        assert product_geometry(main) == unzoomed
        assert value.to_expression() == original and value.structure == structure
    finally:
        page.evaluate("dispose()")
        viewer._on_message(widget, page.evaluate("messages.at(-1)"), [])
        viewer.close()


def test_expanded_product_uses_full_width_of_wrapping_sum(mathml_browser_page):
    source = E("a" * 36 + "+" + "*".join(f"b{index}" for index in range(18)))
    original = bytes(source.__getstate__())
    viewer = source.mathml_viewer()
    compact = viewer.page()["layouts"][-1]
    path = next(
        path for path, label in compact["markers"] if "product" in label.lower()
    )
    viewer.handle_message(
        {"action": "expand", "value": path, "layout": compact["layout"], "size": 25}
    )
    page = mathml_browser_page
    widget, host = mount_viewer(page, viewer)
    viewer._on_message(widget, page.evaluate("messages[0]"), [])
    main = host.locator(".math-content")
    try:
        measurements = {}
        for width in (800, 350, 800):
            host.evaluate(
                "(element, width) => element.style.width = `${width}px`", width
            )
            settle_layout(page)
            product = product_geometry(main)
            assert len(product["factors"]) == 18, product
            assert main.locator('mrow[style*="flex-wrap:wrap"]').count()
            if width in measurements:
                assert product == measurements[width]
            measurements[width] = product
        wide, narrow = measurements[800], measurements[350]
        # The other sum term may wrap onto its own line. Its width must not be
        # subtracted from every row of the product.
        assert sum(factor["width"] for factor in wide["factors"]) < 750, wide
        assert wide["rows"] == 1 and narrow["rows"] > 1, (wide, narrow)
        assert [factor["html"] for factor in wide["factors"]] == [
            factor["html"] for factor in narrow["factors"]
        ]
        assert bytes(source.__getstate__()) == original
    finally:
        page.evaluate("dispose()")
        viewer._on_message(widget, page.evaluate("messages.at(-1)"), [])
        viewer.close()


@pytest.mark.parametrize("kind", ("tensor", "expression"))
def test_fraction_geometry_and_product_height(mathml_browser_page, kind):
    rep = Representation.mink(4)
    labels = ["mu", "nu", *(f"i{i}" for i in range(8))]
    value = TensorName("fraction_page::T0")(rep(labels[0]))
    for index, label in enumerate(labels[1:], 1):
        value *= TensorName(f"fraction_page::T{index}")(rep(label))
    value /= 8
    original, structure = value.to_expression(), value.structure
    viewer = value.paged() if kind == "tensor" else original.mathml_viewer()
    state = viewer.page()
    for candidate in (state, *state["layouts"]):
        assert "@font-face" not in candidate["html"]
    page = mathml_browser_page
    widget, host = mount_viewer(page, viewer)
    viewer._on_message(widget, page.evaluate("messages[0]"), [])
    main = host.locator(".math-content")
    try:
        for width in (1400, 350):
            host.evaluate(
                "(element, width) => element.style.width = `${width}px`", width
            )
            page.evaluate(
                """async () => {
                await new Promise(requestAnimationFrame);
                await new Promise(requestAnimationFrame);
            }"""
            )
            geometry = mathml_geometry(main)
            assert_stacked_fractions(geometry)
            assert geometry["scrollWidth"] <= geometry["availableWidth"] + 2, geometry
            if width == 1400 and kind == "tensor":
                # Ten distinct semantic leaves have fresh free indices. Their
                # short tensor notation fits together in the wide host; pretty
                # MathML whitespace must not turn it into a ten-line column.
                assert geometry["math"]["height"] <= 6 * geometry["fontSize"], geometry
            if kind == "tensor":
                assert "mu" in main.inner_text() and "nu" in main.inner_text()
            assert main.locator(":scope > math").count() == 1
        assert value.to_expression() == original and value.structure == structure
    finally:
        page.evaluate("dispose()")
        viewer._on_message(widget, page.evaluate("messages.at(-1)"), [])
        viewer.close()


def test_shared_widget_hover_paging_stale_previews_zoom_and_disposal(browser_page):
    rep = Representation.mink(4)
    tensor = TensorName("frontend_page::T")(rep("mu"), rep("nu"))
    value = sum(E(f"frontend_page::x^{i}") * tensor for i in range(201))
    original, structure = value.to_expression(), value.structure
    viewer = value.paged(page_size=25)
    assert type(viewer) is MathMLViewer

    page = browser_page
    widget, host = mount_viewer(page, viewer)
    main = host.locator(".math-content")

    def main_html():
        # Preview accessibility state changes without changing the math.
        return main.evaluate("""element => {
            const copy = element.cloneNode(true);
            for (const node of copy.querySelectorAll('button.omission')) {
                for (const name of ['aria-describedby', 'aria-expanded', 'aria-controls']) {
                    node.removeAttribute(name);
                }
            }
            return copy.innerHTML;
        }""")

    page.evaluate("async () => { await document.fonts.ready; }")
    first_html = main_html()
    assert page.evaluate("callbackCount()") == 2
    attach = page.evaluate("messages[0]")
    viewer._on_message(widget, attach, [])
    assert attach["action"] == "attach"
    assert main.locator("math").count() == 1
    assert main.locator(":scope > math").count() == 1
    assert main.locator(":scope > div[data-spenso-math]").count() == 0
    assert '"STIX Two Math"' in main.locator("math").evaluate(
        "element => getComputedStyle(element).fontFamily"
    )
    assert "mu" in main.inner_text() and "nu" in main.inner_text()

    # Hover asks for only the omitted tail, while keeping the main page intact.
    marker = main.locator("button.omission").last
    marker.hover()
    page.wait_for_function("messages.at(-1).action === 'preview'")
    request = page.evaluate("messages.at(-1)")
    field, hover = viewer._message(request)
    assert field == "preview" and not request["popup"]
    assert hover["start"] == viewer._selected_page({"layout": request["layout"]})["end"]
    assert hover["end"] - hover["start"] <= 5
    page.evaluate("setField", [field, hover])
    tooltip = host.get_by_role("tooltip")
    assert tooltip.is_visible()
    assert "mu" in tooltip.inner_text() and "nu" in tooltip.inner_text()
    assert main_html() == first_html

    marker.click()
    request = page.evaluate("messages.at(-1)")
    assert request["action"] == "preview" and request["popup"]
    field, popup = viewer._message(request)
    dialog = host.get_by_role("dialog")
    assert marker.get_attribute("aria-expanded") == "true"
    assert marker.get_attribute("aria-controls") == dialog.get_attribute("id")
    # A delayed hover response must not replace a newer pinned popup.
    page.evaluate("setField", ["preview", hover])
    assert dialog.get_attribute("aria-busy") == "true"
    assert "Loading preview" in dialog.inner_text()
    page.evaluate("setField", [field, popup])
    next_button = dialog.get_by_role("button", name="Next", exact=True)
    for _ in range(4):
        before = dialog.locator(".preview-math").inner_html()
        next_button.click()
        assert next_button.is_disabled()
        request = page.evaluate("messages.at(-1)")
        field, following = viewer._message(request)
        page.evaluate("setField", ["preview", popup])
        assert next_button.is_disabled()
        assert dialog.locator(".preview-math").inner_html() == before
        page.evaluate("setField", [field, following])
        assert dialog.locator(".preview-math math").count() == 1
        assert dialog.locator(".preview-math math *").count() < 10_000
        assert "mu" in dialog.inner_text() and "nu" in dialog.inner_text()
        assert main_html() == first_html
        popup = following

    dialog.get_by_role("button", name="Inspect", exact=True).click()
    request = page.evaluate("messages.at(-1)")
    assert request["action"] == "open"
    field, inspected = viewer._message(request)
    page.evaluate("setField", [field, inspected])
    assert inspected["back"]
    host.get_by_role("button", name="Next", exact=True).click()
    page.evaluate("setField", viewer._message(page.evaluate("messages.at(-1)")))
    assert main.locator("math").count() == 1
    host.get_by_role("button", name="Back to parent").click()
    page.evaluate("setField", viewer._message(page.evaluate("messages.at(-1)")))
    assert main_html() == first_html

    expression = host.get_by_role("region", name="Expression", exact=True)
    expression.focus()
    expression.press("=")
    assert main.evaluate("e=>e.style.getPropertyValue('--expression-zoom')") == "1.25"
    expression.press("-")
    assert main.evaluate("e=>e.style.getPropertyValue('--expression-zoom')") == "1"
    expression.press("=")
    expression.press("0")
    assert main.evaluate("e=>e.style.getPropertyValue('--expression-zoom')") == "1"

    page.evaluate("dispose()")
    assert page.evaluate("callbackCount()") == 0
    dispose = page.evaluate("messages.at(-1)")
    assert dispose == {"action": "dispose", "view": attach["view"]}
    viewer._on_message(widget, dispose, [])
    assert viewer._widget is None
    assert host.locator("math").count() == 0
    assert value.to_expression() == original and value.structure == structure
    viewer.close()


def test_responsive_width_fit_and_inline_expansion_keep_tensor_notation(browser_page):
    rep = Representation.mink(4)
    tensor = TensorName("responsive_page::T")(rep("mu"), rep("nu"))
    left = E("+".join(f"left{i:02}" for i in range(18)))
    right = E("+".join(f"right{i:02}" for i in range(19)))
    value = tensor * left * right
    original, structure = value.to_expression(), value.structure
    viewer = value.paged()
    page = browser_page
    widget, host = mount_viewer(page, viewer)
    viewer._on_message(widget, page.evaluate("messages[0]"), [])
    main = host.locator(".math-content")
    host.evaluate("element => { element.style.width = '1400px'; }")
    page.wait_for_function(
        """() => {
        const content = document.querySelector('#viewer').shadowRoot.querySelector('div').shadowRoot.querySelector('.math-content');
        return content.dataset.layout !== undefined;
    }"""
    )
    wide_layout = int(main.get_attribute("data-layout"))
    wide_requests = page.evaluate("messages.length")
    host.evaluate("element => { element.style.width = '350px'; }")
    page.wait_for_function(
        """wide => {
        const content = document.querySelector('#viewer').shadowRoot.querySelector('div').shadowRoot.querySelector('.math-content');
        return Number(content.dataset.layout) > wide;
    }""",
        arg=wide_layout,
    )
    assert page.evaluate("messages.length") == wide_requests
    assert main.evaluate(
        "element => element.scrollWidth <= element.parentElement.clientWidth + 2"
    )
    assert "mu" in main.inner_text() and "nu" in main.inner_text()
    assert main.locator("math").count() == 1
    assert main.locator("[data-symbolica-path]").count() > 1
    assert main.locator(":scope > math").count() == 1
    assert main.locator(":scope > div[data-spenso-math]").count() == 0

    marker = main.get_by_role("button", name=re.compile(r"^Open sum")).first
    before = main.locator("math *").count()
    marker.click()
    request = page.evaluate("messages.at(-1)")
    assert request["action"] == "preview" and request["popup"]
    page.evaluate("setField", viewer._message(request))
    dialog = host.get_by_role("dialog")
    expand = dialog.get_by_role("button", name="Show more here", exact=True)
    assert expand.is_visible() and expand.is_enabled()
    expand.click()
    request = page.evaluate("messages.at(-1)")
    assert request["action"] == "expand"
    field, expanded = viewer._message(request)
    assert field == "page" and "error" not in expanded
    page.evaluate("setField", [field, expanded])
    assert main.locator("math *").count() > before
    assert not expanded["back"]
    assert "mu" in main.inner_text() and "nu" in main.inner_text()
    assert main.locator("math").count() == 1
    assert main.locator("math *").count() <= 10_000
    assert value.to_expression() == original and value.structure == structure
    page.evaluate("dispose()")
    viewer._on_message(widget, page.evaluate("messages.at(-1)"), [])
    viewer.close()
